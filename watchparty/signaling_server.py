"""Simple HTTP signaling server for Watch‑Party.

The server runs in a background ``threading.Thread`` and stores pending offers and
answers in an in‑memory dictionary. It exposes four endpoints:

- ``POST /offer`` – host sends its SDP offer JSON. The server generates a random
  ``session_id`` and stores the offer.
- ``GET /offer/<session_id>`` – guest polls for the offer.
- ``POST /answer/<session_id>`` – guest sends its SDP answer JSON.
- ``GET /answer/<session_id>`` – host polls for the answer.

The server is deliberately lightweight – it uses ``http.server`` from the
standard library and does not write any files to disk. It stops automatically
when ``shutdown()`` is called (e.g., when the watch‑party ends).
"""
import json
import logging
logger = logging.getLogger(__name__)

import threading
import uuid
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


class _SignalingHandler(BaseHTTPRequestHandler):
    # Shared state across all handler instances (class attribute)
    _store = {
        "offers": {},   # session_id -> offer JSON string
        "answers": {},  # session_id -> answer JSON string
    }

    def _set_json(self, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode("utf-8") if length else ""

    def do_POST(self):
        parsed = urlparse(self.path)
        logger.debug("POST request path: %s", parsed.path)
        if parsed.path == "/offer":
            # Host creates a new session – generate an ID and store the offer.
            body = self._read_body()
            try:
                offer = json.loads(body)
            except json.JSONDecodeError:
                self._set_json(400)
                self.wfile.write(b"{\"error\": \"invalid JSON\"}")
                logger.warning("Invalid JSON in /offer POST")
                return
            session_id = str(uuid.uuid4())
            self._store["offers"][session_id] = offer
            self._set_json(200)
            self.wfile.write(json.dumps({"session_id": session_id}).encode())
            logger.info("Stored offer for session %s", session_id)
            return
        if parsed.path.startswith("/answer/"):
            # Guest posts answer for a known session.
            session_id = parsed.path.split("/")[-1]
            if session_id not in self._store["offers"]:
                self._set_json(404)
                self.wfile.write(b"{\"error\": \"session not found\"}")
                logger.warning("Answer POST for unknown session %s", session_id)
                return
            body = self._read_body()
            try:
                answer = json.loads(body)
            except json.JSONDecodeError:
                self._set_json(400)
                self.wfile.write(b"{\"error\": \"invalid JSON\"}")
                logger.warning("Invalid JSON in /answer POST for session %s", session_id)
                return
            self._store["answers"][session_id] = answer
            self._set_json(200)
            self.wfile.write(b"{\"status\": \"ok\"}")
            logger.info("Stored answer for session %s", session_id)
            return
        # Unknown endpoint
        self._set_json(404)
        self.wfile.write(b"{\"error\": \"unknown endpoint\"}")
        logger.warning("POST to unknown endpoint: %s", parsed.path)

    def do_GET(self):
        parsed = urlparse(self.path)
        logger.debug("GET request path: %s", parsed.path)
        if parsed.path.startswith("/offer/"):
            # Guest polls for the offer.
            session_id = parsed.path.split("/")[-1]
            offer = self._store["offers"].get(session_id)
            if offer is None:
                self._set_json(404)
                self.wfile.write(b"{\"error\": \"offer not found\"}")
                logger.warning("GET offer not found for session %s", session_id)
                return
            self._set_json(200)
            self.wfile.write(json.dumps(offer).encode())
            logger.info("Returned offer for session %s", session_id)
            return
        if parsed.path.startswith("/answer/"):
            # Host polls for the answer.
            session_id = parsed.path.split("/")[-1]
            answer = self._store["answers"].get(session_id)
            if answer is None:
                self._set_json(404)
                self.wfile.write(b"{\"error\": \"answer not found\"}")
                logger.warning("GET answer not found for session %s", session_id)
                return
            self._set_json(200)
            self.wfile.write(json.dumps(answer).encode())
            logger.info("Returned answer for session %s", session_id)
            return
        self._set_json(404)
        self.wfile.write(b"{\"error\": \"unknown endpoint\"}")
        logger.warning("GET to unknown endpoint: %s", parsed.path)

    def log_message(self, format, *args):
        # Suppress standard HTTP server logging – the app already has its own
        # logging facilities.
        return


class SignalingServer:
    """Encapsulates the HTTP server running in a background thread.

    The ``host`` and ``port`` are configurable; by default it binds to
    ``127.0.0.1:0`` which lets the OS pick an available port. The actual listening
    address can be retrieved via ``self.url`` after ``start()``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        advertised_host: str | None = None,
    ):
        self.server = HTTPServer((host, port), _SignalingHandler)
        self.thread: threading.Thread | None = None
        if advertised_host is None:
            advertised_host = self._detect_advertised_host()
        self.url = f"http://{advertised_host}:{self.server.server_address[1]}"

    def start(self) -> None:
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()
        if self.thread:
            self.thread.join()

    def _detect_advertised_host(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
