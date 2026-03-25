"""Watch‑Party core manager.

This module defines the ``WatchPartySession`` class that holds the state of a
watch‑party (participants, current media, sync status) and provides simple
methods to update media, broadcast sync packets, and handle participant
join/leave.  The implementation is deliberately minimal – the real networking
logic lives in the ``watchparty_ui`` integration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

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

    def end_session(self) -> None:
        """Mark the session as ended and clear participants."""
        self.is_active = False
        self.participants.clear()

# End of file
