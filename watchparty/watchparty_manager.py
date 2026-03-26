"""Watch‑Party core manager.

This module defines the ``WatchPartySession`` class that holds the state of a
watch‑party (participants, current media, sync status) and provides simple
methods to update media, broadcast sync packets, and handle participant
join/leave.  The implementation is deliberately minimal – the real networking
logic lives in the ``watchparty_ui`` integration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import threading
import time
import logging
logger = logging.getLogger(__name__)

import requests
from .signaling_server import SignalingServer
from .aiortc_handler import AiortcHandler, run_async
import json


@dataclass
class MediaInfo:
    """Simple container for the media being watched."""
    url: str
    title: str = ""
    start: float = 0.0  # start position in seconds

@dataclass
class ParticipantInfo:
    """Tracks a participant's identifier and connection state."""
    nick: str
    # In a full implementation this would hold the WebRTC data‑channel.
    # Here we keep it generic.
    connection: Any = None

@dataclass
class WatchPartySession:
    """Central session state for a watch‑party.

    Attributes
    ----------
    session_id: str
        Unique identifier for the session (e.g., UUID).
    participants: Dict[str, ParticipantInfo]
        Mapping nickname → participant info.
    current_media: MediaInfo
        Media currently being streamed.
    is_active: bool
        Flag indicating whether the party is ongoing.
    """

    session_id: str
    participants: Dict[str, ParticipantInfo] = field(default_factory=dict)
    current_media: MediaInfo = field(default_factory=lambda: MediaInfo(url=""))
    is_active: bool = True
    # Additional runtime fields
    role: str | None = None  # "host" or "guest"
    handler: AiortcHandler | None = None
    signaling: SignalingServer | None = None
    on_chat_message: Any = None
    _host_poll_thread: threading.Thread | None = None
    _guest_poll_thread: threading.Thread | None = None

    def add_participant(self, nick: str, connection: Any = None) -> None:
        """Register a new participant.
        If the nick already exists it is overwritten.
        """
        self.participants[nick] = ParticipantInfo(nick=nick, connection=connection)

    def remove_participant(self, nick: str) -> None:
        """Remove a participant from the session."""
        self.participants.pop(nick, None)

    def update_media(self, url: str, title: str = "", start: float = 0.0) -> None:
        """Replace the current media information.
        This method would also trigger a ``media_change`` broadcast in the
        networking layer.
        """
        self.current_media = MediaInfo(url=url, title=title, start=start)

    def broadcast_sync(self, timestamp: float) -> None:
        """Placeholder for broadcasting an NTP sync packet to all participants.
        The real implementation uses the WebRTC data‑channel.
        """
        # In a full implementation we would iterate over participants and
        # send a JSON payload containing the timestamp.
        pass

    def start_host(self) -> str:
        """Initialize the session as host, start signaling server, create offer.
        Returns the URL that guests should use to join (includes the session ID).
        """
        self.role = "host"
        logger.info("Starting host session")
        # Start local signaling server.
        self.signaling = SignalingServer()
        self.signaling.start()
        logger.debug("Signaling server started at %s", self.signaling.url)
        # Create aiortc handler.
        self.handler = AiortcHandler(is_host=True, on_message=self._handle_incoming)
        logger.debug("AiortcHandler created for host")
        # Generate SDP offer.
        offer_json = run_async(self.handler.create_offer())
        logger.debug("Watchparty offer created: %s", offer_json)
        # Convert JSON string to dict before sending to signaling server.
        offer_dict = json.loads(offer_json)
        # POST offer to server.
        resp = requests.post(f"{self.signaling.url}/offer", json=offer_dict)
        resp.raise_for_status()
        logger.debug("Watchparty offer posted to signaling server")
        session_id = resp.json()["session_id"]
        self.session_id = session_id
        logger.info("Host session created with ID %s", session_id)
        # Start background thread to poll for answer.
        def poll_answer():
            while self.is_active:
                try:
                    r = requests.get(f"{self.signaling.url}/answer/{session_id}")
                    if r.status_code == 200:
                        answer = r.json()
                        logger.debug("Watchparty answer fetched from signaling server: %s", answer)
                        run_async(self.handler.set_remote(answer))
                        logger.debug("Watchparty remote answer applied on host")
                        break
                except Exception as e:
                    logger.exception("Error polling for answer: %s", e)
                    raise
                time.sleep(0.5)
        self._host_poll_thread = threading.Thread(target=poll_answer, daemon=True)
        self._host_poll_thread.start()
        # Return full URL for guests.
        return f"{self.signaling.url}/offer/{session_id}"

    def join_host(self, offer_url: str) -> None:
        """Join an existing host using the URL provided by the host.
        The method fetches the SDP offer, creates an answer, and posts it back.
        """
        self.role = "guest"
        logger.info("Joining host with URL %s", offer_url)
        # Extract base URL and session ID.
        parts = offer_url.rstrip("/").split("/")
        session_id = parts[-1]
        base_url = "/".join(parts[:-2])  # drop '/offer/<id>'
        self.handler = AiortcHandler(is_host=False, on_message=self._handle_incoming)
        logger.debug("AiortcHandler created for guest")
        try:
            # Retrieve the offer.
            r = requests.get(offer_url)
            r.raise_for_status()
            offer = r.json()
            logger.debug("Watchparty offer fetched from host: %s", offer)
            # Create answer; create_answer() applies the remote offer internally.
            answer = run_async(self.handler.create_answer(offer))
            answer_payload = json.loads(answer)
            logger.debug("Watchparty answer created: %s", answer)
            # POST answer back to host.
            resp = requests.post(f"{base_url}/answer/{session_id}", json=answer_payload)
            resp.raise_for_status()
            logger.debug("Watchparty answer posted to host for session %s", session_id)
            # No further polling needed – connection established.
        except Exception as e:
            raise

    def _handle_incoming(self, data: dict) -> None:
        """Dispatch incoming JSON messages to appropriate session actions.
        Expected keys: ``type`` ("chat", "sync", "media_change").
        """
        msg_type = data.get("type")
        if msg_type == "chat":
            self._last_chat = data.get("msg")
            if callable(self.on_chat_message):
                self.on_chat_message(self._last_chat)
        elif msg_type == "sync":
            self._last_sync = data.get("ts")
        elif msg_type == "media_change":
            self.current_media = MediaInfo(
                url=data.get("url", ""),
                title=data.get("title", ""),
                start=data.get("start", 0.0),
            )
        # Additional message types can be added later.


    def end_session(self) -> None:
        """Mark the session as ended, clear participants, and clean up resources."""
        self.is_active = False
        self.participants.clear()
        # Clean up aiortc handler
        if self.handler:
            run_async(self.handler.close())
            self.handler = None
        # Shut down signaling server if we started one
        if self.signaling:
            self.signaling.shutdown()
            self.signaling = None
        # Reset polling threads
        self._host_poll_thread = None
        self._guest_poll_thread = None

# End of file
