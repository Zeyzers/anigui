"""Watch‑Party UI integration.

This file provides a lightweight UI for the Watch‑Party feature.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Signal, Slot
import logging
logger = logging.getLogger(__name__)

class WatchPartyTab(QWidget):
    chat_received = Signal(str)
    playback_received = Signal(object)
    media_change_received = Signal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.chat_received.connect(self._append_remote_message)
        self.session.on_chat_message = self.chat_received.emit
        self.session.on_playback_event = self.playback_received.emit
        self.session.on_media_change = self.media_change_received.emit
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel("Watch‑Party Session")
        layout.addWidget(self.title_label)
        self.info = QLabel("No participants yet.")
        layout.addWidget(self.info)
        # Session control buttons
        self.btn_create = QPushButton("Create Party (Host)")
        self.btn_create.clicked.connect(self.create_party)
        layout.addWidget(self.btn_create)
        self.btn_join = QPushButton("Join Party (Guest)")
        self.btn_join.clicked.connect(self.join_party)
        layout.addWidget(self.btn_join)
        self.url_field = QLineEdit()
        self.url_field.setPlaceholderText(
            "Session URL (auto‑filled for host, paste for guest)"
        )
        layout.addWidget(self.url_field)
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        layout.addWidget(self.chat)
        self.input = QTextEdit()
        self.input.setFixedHeight(50)
        layout.addWidget(self.input)
        self.send_btn = QPushButton("Send")
        layout.addWidget(self.send_btn)
        self.send_btn.clicked.connect(self.send_message)

    def send_message(self):
        msg = self.input.toPlainText().strip()
        if not msg:
            return
        self.chat.append(f"You: {msg}")
        self.input.clear()
        has_open_guest_channel = False
        if getattr(self.session, "handler", None) is not None:
            ch = getattr(self.session.handler, "channel", None)
            has_open_guest_channel = bool(ch and getattr(ch, "readyState", None) == "open")
        has_open_host_peer = bool(self.session._connected_host_handlers())
        if not (has_open_guest_channel or has_open_host_peer):
            logger.warning("Data channel not open when sending message")
            QMessageBox.warning(self, "Error", "Data channel not established yet")
            return
        try:
            self.session.send_chat(msg)
        except Exception as e:
            logger.exception("Failed to send chat message")
            QMessageBox.warning(self, "Error", f"Failed to send message: {e}")

    @Slot(str)
    def _append_remote_message(self, msg: str) -> None:
        self.chat.append(f"Peer: {msg}")

    def create_party(self):
        """Host creates a new party – start signaling and generate URL."""
        try:
            url = self.session.start_host()
            self.url_field.setText(url)
            self.info.setText("Waiting for guests to join…")
        except Exception as e:
            QMessageBox.critical(self, "Host Error", str(e))

    def join_party(self):
        """Guest joins an existing party using the URL pasted in the field."""
        url = self.url_field.text().strip()
        if not url:
            QMessageBox.warning(
                self, "Input required", "Paste the session URL provided by the host."
            )
            return
        try:
            self.session.join_host(url)
            self.info.setText("Connected to host. Ready to chat and sync.")
        except Exception as e:
            logger.exception("Watchparty guest join failed while fetching offer, creating answer, or posting answer")
            QMessageBox.critical(self, "Join Error", str(e))
