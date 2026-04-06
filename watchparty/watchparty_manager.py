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
import uuid
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
    client_id: str | None = None
    host_handlers: dict[str, AiortcHandler] = field(default_factory=dict)
    on_chat_message: Any = None
    on_playback_event: Any = None
    on_media_change: Any = None
    get_host_state_snapshot: Any = None
    _host_poll_thread: threading.Thread | None = None
    _host_join_poll_thread: threading.Thread | None = None
    _host_answer_poll_threads: dict[str, threading.Thread] = field(default_factory=dict)
    _guest_poll_thread: threading.Thread | None = None
    _host_handlers_lock: threading.RLock = field(default_factory=threading.RLock)
    _initial_state_synced_peers: set[str] = field(default_factory=set)

    def _host_debug_state(self) -> dict[str, Any]:
        with self._host_handlers_lock:
            handler_items = list(self.host_handlers.items())
        return {
            "host_handlers": [peer_id for peer_id, _ in handler_items],
            "host_answer_poll_threads": list(self._host_answer_poll_threads.keys()),
            "initial_state_synced_peers": list(self._initial_state_synced_peers),
            "peer_states": [
                {
                    "peer_id": peer_id,
                    "handler_exists": handler is not None,
                    "channel_exists": getattr(handler, "channel", None) is not None,
                    "ready_state": getattr(getattr(handler, "channel", None), "readyState", None),
                }
                for peer_id, handler in handler_items
            ],
        }

    def _connected_host_handlers(self) -> list[AiortcHandler]:
        out: list[AiortcHandler] = []
        with self._host_handlers_lock:
            handlers = list(self.host_handlers.values())
        for handler in handlers:
            ch = getattr(handler, "channel", None)
            if ch is not None and getattr(ch, "readyState", None) == "open":
                out.append(handler)
        return out

    def _create_host_peer(self, peer_id: str) -> None:
        with self._host_handlers_lock:
            if peer_id in self.host_handlers or not self.signaling:
                logger.info(
                    "Watchparty host peer creation skipped peer_id=%s signaling=%s existing_peers=%s",
                    peer_id,
                    self.signaling is not None,
                    list(self.host_handlers.keys()),
                )
                return
        print(f"HOST CREATING HANDLER peer_id={peer_id!r}", flush=True)
        handler = AiortcHandler(
            is_host=True,
            on_message=self._handle_incoming,
            on_disconnect=lambda: self._cleanup_host_peer(peer_id, close_handler=False),
            on_channel_open=lambda: self._sync_peer_initial_state(peer_id),
        )
        with self._host_handlers_lock:
            self.host_handlers[peer_id] = handler
        logger.info("Watchparty host handler created for peer_id=%s", peer_id)
        print(f"HOST CREATING OFFER peer_id={peer_id!r}", flush=True)
        offer_json = run_async(handler.create_offer())
        logger.debug("Watchparty offer created for peer %s: %s", peer_id, offer_json)
        offer_dict = json.loads(offer_json)
        print(
            f"HOST POSTING OFFER session_id={self.session_id!r} peer_id={peer_id!r}",
            flush=True,
        )
        resp = requests.post(
            f"{self.signaling.url}/offer/{self.session_id}/{peer_id}",
            json=offer_dict,
        )
        resp.raise_for_status()
        print(
            f"HOST POSTED OFFER session_id={self.session_id!r} peer_id={peer_id!r} status={resp.status_code}",
            flush=True,
        )
        self._start_host_answer_poll(peer_id)

    def _start_host_answer_poll(self, peer_id: str) -> None:
        if not self.signaling or peer_id in self._host_answer_poll_threads:
            return

        def poll_answer() -> None:
            try:
                while self.is_active:
                    with self._host_handlers_lock:
                        handler = self.host_handlers.get(peer_id)
                    if handler is None:
                        break
                    r = requests.get(f"{self.signaling.url}/answer/{self.session_id}/{peer_id}")
                    if r.status_code == 200:
                        answer = r.json()
                        logger.debug(
                            "Watchparty answer fetched from signaling server for peer %s: %s",
                            peer_id,
                            answer,
                        )
                        run_async(handler.set_remote(answer))
                        logger.debug(
                            "Watchparty remote answer applied on host for peer %s",
                            peer_id,
                        )
                        self._sync_peer_initial_state(peer_id)
                        break
                    if r.status_code != 404:
                        r.raise_for_status()
                    time.sleep(0.5)
            except Exception as e:
                logger.exception("Error polling answer for peer %s: %s", peer_id, e)
                raise
            finally:
                self._host_answer_poll_threads.pop(peer_id, None)

        thread = threading.Thread(target=poll_answer, daemon=True)
        self._host_answer_poll_threads[peer_id] = thread
        thread.start()

    def _send_payload_to_peer(self, peer_id: str, payload: dict[str, Any]) -> None:
        with self._host_handlers_lock:
            handler = self.host_handlers.get(peer_id)
        logger.info(
            "Watchparty targeted send peer_id=%s payload_type=%s handler_exists=%s channel_exists=%s ready_state=%s host_state=%s",
            peer_id,
            payload.get("type"),
            handler is not None,
            getattr(handler, "channel", None) is not None if handler is not None else False,
            getattr(getattr(handler, "channel", None), "readyState", None) if handler is not None else None,
            self._host_debug_state(),
        )
        if handler is None:
            return False
        ch = getattr(handler, "channel", None)
        if ch is None or getattr(ch, "readyState", None) != "open":
            return False
        try:
            run_async(handler.send(payload))
        except Exception:
            logger.exception(
                "Watchparty targeted send crashed peer_id=%s payload=%s failing_line=_send_payload_to_peer:run_async(handler.send(payload))",
                peer_id,
                payload,
            )
            raise
        return True

    def _sync_peer_initial_state(self, peer_id: str) -> None:
        if peer_id in self._initial_state_synced_peers:
            return
        if not callable(self.get_host_state_snapshot):
            return
        try:
            snapshot = self.get_host_state_snapshot()
        except Exception:
            logger.exception("Failed to build host state snapshot for peer %s", peer_id)
            return
        if not isinstance(snapshot, dict):
            return
        media_payload = snapshot.get("media_change")
        playback_payload = snapshot.get("playback")
        sent_any = False
        if isinstance(media_payload, dict):
            sent_any = self._send_payload_to_peer(peer_id, media_payload) or sent_any
        if isinstance(playback_payload, dict):
            sent_any = self._send_payload_to_peer(peer_id, playback_payload) or sent_any
        if sent_any:
            self._initial_state_synced_peers.add(peer_id)

    def _cleanup_host_peer(self, peer_id: str, *, close_handler: bool = True) -> None:
        logger.info(
            "Watchparty cleanup removing peer_id=%s close_handler=%s pre_state=%s",
            peer_id,
            close_handler,
            self._host_debug_state(),
        )
        with self._host_handlers_lock:
            handler = self.host_handlers.pop(peer_id, None)
        self._host_answer_poll_threads.pop(peer_id, None)
        self._initial_state_synced_peers.discard(peer_id)
        if self.signaling:
            self.signaling.remove_peer(self.session_id, peer_id)
        if close_handler and handler is not None:
            try:
                run_async(handler.close())
            except Exception:
                logger.exception("Failed to close host peer handler for %s", peer_id)
        logger.info(
            "Cleaned up host peer %s post_state=%s",
            peer_id,
            self._host_debug_state(),
        )

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

    def _send_payload(self, payload: dict[str, Any]) -> None:
        if self.role == "host":
            host_state = self._host_debug_state()
            logger.info(
                "Watchparty host broadcast start payload_type=%s host_state=%s",
                payload.get("type"),
                host_state,
            )
            try:
                with self._host_handlers_lock:
                    handler_items = list(self.host_handlers.items())
                logger.info(
                    "Watchparty host broadcast targets peer_ids=%s",
                    [peer_id for peer_id, _ in handler_items],
                )
                for peer_id, handler in handler_items:
                    ch = getattr(handler, "channel", None)
                    logger.info(
                        "Watchparty host broadcast peer peer_id=%s handler_exists=%s channel_exists=%s ready_state=%s",
                        peer_id,
                        handler is not None,
                        ch is not None,
                        getattr(ch, "readyState", None) if ch is not None else None,
                    )
                    if ch is not None and getattr(ch, "readyState", None) == "open":
                        run_async(handler.send(payload))
            except Exception:
                logger.exception(
                    "Watchparty host broadcast crashed payload=%s host_state=%s failing_line=_send_payload:run_async(handler.send(payload))",
                    payload,
                    self._host_debug_state(),
                )
                raise
            return
        if self.role == "guest" and self.handler and self.handler.channel:
            if getattr(self.handler.channel, "readyState", None) == "open":
                run_async(self.handler.send(payload))

    def send_chat(self, msg: str) -> None:
        self._send_payload({"type": "chat", "msg": msg})

    def broadcast_playback(
        self,
        action: str,
        position: float | None = None,
        ratio: float | None = None,
    ) -> None:
        if self.role != "host":
            return
        payload: dict[str, Any] = {"type": "playback", "action": action}
        if position is not None:
            payload["position"] = float(position)
        if ratio is not None:
            payload["ratio"] = float(ratio)
        self._send_payload(payload)

    def broadcast_media_change(
        self,
        media_id: str,
        provider: str,
        identifier: str,
        lang_name: str,
        name: str,
        episode: float | int,
        position: float | None = None,
        playing: bool | None = None,
    ) -> None:
        if self.role != "host":
            return
        payload: dict[str, Any] = {
            "type": "media_change",
            "media_id": media_id,
            "provider": provider,
            "identifier": identifier,
            "lang": lang_name,
            "name": name,
            "episode": episode,
        }
        if position is not None:
            payload["position"] = float(position)
        if playing is not None:
            payload["playing"] = bool(playing)
        self._send_payload(payload)

    def start_host(self) -> str:
        """Initialize the session as host and return the guest join URL."""
        self.role = "host"
        logger.info("Starting host session")
        self.signaling = SignalingServer()
        self.signaling.start()
        logger.debug("Signaling server started at %s", self.signaling.url)
        resp = requests.post(f"{self.signaling.url}/session")
        resp.raise_for_status()
        self.session_id = resp.json()["session_id"]
        logger.info("Host session created with ID %s", self.session_id)

        def poll_joins():
            while self.is_active:
                try:
                    print(f"HOST JOIN POLL TICK session_id={self.session_id!r}", flush=True)
                    r = requests.get(f"{self.signaling.url}/join/{self.session_id}")
                    if r.status_code == 200:
                        peer_ids = r.json().get("peer_ids", [])
                        print(f"HOST JOIN POLL RECEIVED peer_ids={peer_ids!r}", flush=True)
                        for peer_id in peer_ids:
                            self._create_host_peer(str(peer_id))
                    else:
                        logger.warning(
                            "Watchparty host join poll non-200 session_id=%s status=%s",
                            self.session_id,
                            r.status_code,
                        )
                except Exception as e:
                    logger.exception("Error polling for join requests: %s", e)
                    raise
                time.sleep(0.5)

        self._host_join_poll_thread = threading.Thread(target=poll_joins, daemon=True)
        self._host_join_poll_thread.start()
        return f"{self.signaling.url}/session/{self.session_id}"

    def join_host(self, offer_url: str) -> None:
        """Join an existing host using the URL provided by the host.
        The method fetches the SDP offer, creates an answer, and posts it back.
        """
        self.role = "guest"
        logger.info("Joining host with URL %s", offer_url)
        parts = offer_url.rstrip("/").split("/")
        session_id = parts[-1]
        if len(parts) >= 2 and parts[-2] == "session":
            base_url = "/".join(parts[:-2])
        else:
            base_url = "/".join(parts[:-2])
        self.client_id = str(uuid.uuid4())
        self.handler = AiortcHandler(is_host=False, on_message=self._handle_incoming)
        logger.debug("AiortcHandler created for guest")
        try:
            join_resp = requests.post(
                f"{base_url}/join/{session_id}",
                json={"peer_id": self.client_id},
            )
            join_resp.raise_for_status()

            offer = None
            while self.is_active and offer is None:
                r = requests.get(f"{base_url}/offer/{session_id}/{self.client_id}")
                if r.status_code == 200:
                    offer = r.json()
                    logger.debug(
                        "Watchparty offer fetched from host for peer %s: %s",
                        self.client_id,
                        offer,
                    )
                    break
                if r.status_code != 404:
                    r.raise_for_status()
                time.sleep(0.5)
            if offer is None:
                raise RuntimeError("Timed out waiting for host offer")
            answer = run_async(self.handler.create_answer(offer))
            answer_payload = json.loads(answer)
            logger.debug("Watchparty answer created: %s", answer)
            resp = requests.post(
                f"{base_url}/answer/{session_id}/{self.client_id}",
                json=answer_payload,
            )
            resp.raise_for_status()
            logger.debug(
                "Watchparty answer posted to host for session %s peer %s",
                session_id,
                self.client_id,
            )
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
        elif msg_type == "playback":
            if callable(self.on_playback_event):
                self.on_playback_event(data)
        elif msg_type == "media_change":
            if callable(self.on_media_change):
                self.on_media_change(data)
        elif msg_type == "sync":
            self._last_sync = data.get("ts")
        # Additional message types can be added later.


    def end_session(self) -> None:
        """Mark the session as ended, clear participants, and clean up resources."""
        self.is_active = False
        self.participants.clear()
        if self.handler:
            run_async(self.handler.close())
            self.handler = None
        for peer_id in list(self.host_handlers):
            self._cleanup_host_peer(peer_id)
        if self.signaling:
            self.signaling.shutdown()
            self.signaling = None
        self._host_poll_thread = None
        self._host_join_poll_thread = None
        self._guest_poll_thread = None

# End of file
