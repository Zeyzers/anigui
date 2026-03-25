"""Watch‑Party UI integration.

This file contains stubs that would be imported by ``main_window.py`` to add a
new "Watch‑Party" tab. The actual UI code is intentionally lightweight – the
focus here is to illustrate where the widgets would be placed.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit

class WatchPartyTab(QWidget):
    """Simple placeholder widget for the watch‑party UI.

    In a full implementation this would embed the participant list, a chat pane,
    and a "Switch Episode" button that calls ``WatchPartySession.update_media``.
    """

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
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        layout.addWidget(self.chat)
        self.input = QTextEdit()
        self.input.setFixedHeight(50)
        layout.addWidget(self.input)
        self.send_btn = QPushButton("Send")
        layout.addWidget(self.send_btn)
        # Connect signals (placeholder – real logic goes elsewhere)
        self.send_btn.clicked.connect(self.send_message)

    def send_message(self):
        # Placeholder for sending a chat message via the session's data channel.
        msg = self.input.toPlainText().strip()
        if msg:
            self.chat.append(f"You: {msg}")
            self.input.clear()
            # In a full implementation we would forward ``msg`` to the session.

# End of file
