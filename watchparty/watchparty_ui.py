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
import logging
logger = logging.getLogger(__name__)

from .aiortc_handler import run_async


class WatchPartyTab(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
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
        if not hasattr(self.session, "handler") or not self.session.handler:
            logger.warning("Attempted to send message but session.handler is missing")
            QMessageBox.warning(self, "Error", "Session not connected")
            return
        ch = getattr(self.session.handler, "channel", None)
        if not (ch and getattr(ch, "readyState", None) == "open"):
            logger.warning("Data channel not open when sending message")
            QMessageBox.warning(self, "Error", "Data channel not established yet")
            return
        try:
            run_async(self.session.handler.send({"type": "chat", "msg": msg}))
        except Exception as e:
            logger.exception("Failed to send chat message")
            QMessageBox.warning(self, "Error", f"Failed to send message: {e}")

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
            QMessageBox.critical(self, "Join Error", str(e))
