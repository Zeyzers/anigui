"""aiortc handler for Watch‑Party.

Provides a thin wrapper around ``aiortc.RTCPeerConnection`` that simplifies
creation of a reliable data channel (named ``control``) and the generation
of SDP offers/answers. The wrapper is deliberately minimal – it only exposes
the functionality required by ``WatchPartySession``.
"""

import asyncio
import concurrent.futures
import logging
import threading


logger = logging.getLogger(__name__)

import json

from typing import Callable, Any

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.signaling import object_from_string, object_to_string


class AiortcHandler:
    """Utility class to manage a single WebRTC peer connection.

    Parameters
    ----------
    is_host: bool
        ``True`` if this endpoint will act as the offerer.
    on_message: Callable[[dict], None]
        Callback invoked for each JSON message received on the ``control`` data
        channel. The callback receives the parsed JSON payload.
    """

    def __init__(self, is_host: bool, on_message: Callable[[dict], None]):
        self.is_host = is_host
        self.on_message = on_message
        self.pc: RTCPeerConnection | None = None
        self.channel: RTCDataChannel | None = None
        self._loop_runner = _ensure_watchparty_loop()
        self._loop_runner.submit(self._async_init()).result()
        logger.debug("AiortcHandler initialized, is_host=%s", is_host)

    async def _async_init(self) -> None:
        self.pc = RTCPeerConnection()
        self.pc.on("datachannel", self._on_datachannel)
        self.pc.on(
            "connectionstatechange",
            lambda: self._log_pc_state("connectionState", self.pc.connectionState),
        )
        self.pc.on(
            "iceconnectionstatechange",
            lambda: self._log_pc_state("iceConnectionState", self.pc.iceConnectionState),
        )
        self.pc.on(
            "signalingstatechange",
            lambda: self._log_pc_state("signalingState", self.pc.signalingState),
        )
        if self.is_host:
            self.channel = self.pc.createDataChannel("control", ordered=True)
            self._attach_channel_handlers(self.channel, role="host")
            logger.debug("Watchparty data channel created (host)")

    # ---------------------------------------------------------------------
    # Signaling helpers
    # ---------------------------------------------------------------------
    async def create_offer(self) -> str:
        logger.debug("Creating SDP offer")
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        # Wait for ICE gathering to complete.
        await self._wait_for_ice_gathering()
        logger.debug("SDP offer created")
        return json.dumps({"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type})

    async def create_answer(self, offer_sdp: str) -> str:
        logger.debug("Creating SDP answer")
        offer = RTCSessionDescription(sdp=offer_sdp["sdp"], type=offer_sdp["type"])
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self._wait_for_ice_gathering()
        logger.debug("SDP answer created")
        return json.dumps({"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type})

    async def set_remote(self, remote_sdp: str) -> None:
        logger.debug("Setting remote SDP description")
        remote = RTCSessionDescription(sdp=remote_sdp["sdp"], type=remote_sdp["type"])
        await self.pc.setRemoteDescription(remote)

    # ---------------------------------------------------------------------
    # Data channel handling
    # ---------------------------------------------------------------------
    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        """Callback invoked when a data channel is created by the remote peer.
        For the guest endpoint we receive the ``control`` channel here.
        """
        self.channel = channel
        self._attach_channel_handlers(self.channel, role="guest")
        logger.debug("Watchparty data channel created (guest)")

    def _attach_channel_handlers(self, channel: RTCDataChannel, role: str) -> None:
        channel.on("message", self._handle_message)
        channel.on(
            "open",
            lambda: (
                logger.info("Watchparty data channel open (%s)", role),
            ),
        )
        channel.on("close", lambda: logger.info("Data channel closed (%s)", role))
        channel.on("error", lambda e: logger.error("Data channel error (%s): %s", role, e))

    def _log_pc_state(self, label: str, value: str) -> None:
        logger.info("Watchparty %s=%s", label, value)

    def _handle_message(self, message: str) -> None:
        """Parse a JSON message received from the peer and forward it to the
        user‑provided ``on_message`` callback.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            # Silently ignore malformed payloads – they are not part of the
            # protocol.
            return
        self.on_message(data)

    async def send(self, payload: dict) -> None:
        logger.debug("Watchparty send attempted: %s", payload)
        if self.channel is None:
            raise RuntimeError("Data channel not established yet")
        try:
            self.channel.send(json.dumps(payload))
        except Exception:
            logger.exception("Watchparty send failed")
            raise

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    async def _wait_for_ice_gathering(self) -> None:
        """Wait until ICE gathering is complete. ``aiortc`` signals this via the
        ``icegatheringstate`` attribute changing to ``complete``.
        """
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        """Close the peer connection and clean up resources."""
        await self.pc.close()

# -------------------------------------------------------------------------
# Helper function used by ``WatchPartySession`` – it runs the async coroutine
# in a background event loop for convenience.
# -------------------------------------------------------------------------


class _WatchPartyLoopRunner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            name="WatchPartyLoop",
            daemon=True,
        )
        self._started = threading.Event()
        self.thread.start()
        self._started.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        logger.info("Watchparty event loop thread started")
        self._started.set()
        self.loop.run_forever()

    def submit(self, coro: Any) -> concurrent.futures.Future:
        logger.debug("Watchparty coroutine scheduled: %r", coro)
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


_WATCHPARTY_LOOP_RUNNER: _WatchPartyLoopRunner | None = None


def _ensure_watchparty_loop() -> _WatchPartyLoopRunner:
    global _WATCHPARTY_LOOP_RUNNER
    if _WATCHPARTY_LOOP_RUNNER is None:
        _WATCHPARTY_LOOP_RUNNER = _WatchPartyLoopRunner()
    return _WATCHPARTY_LOOP_RUNNER

def run_async(coro: Any) -> Any:
    """Execute *coro* on the dedicated watchparty event loop and return the result."""
    return _ensure_watchparty_loop().submit(coro).result()

"""End of file
"""
