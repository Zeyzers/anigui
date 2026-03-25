"""aiortc handler for Watch‑Party.

Provides a thin wrapper around ``aiortc.RTCPeerConnection`` that simplifies
creation of a reliable data channel (named ``control``) and the generation
of SDP offers/answers. The wrapper is deliberately minimal – it only exposes
the functionality required by ``WatchPartySession``.
"""

import asyncio
import logging


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
        self.pc = RTCPeerConnection()
        self.channel: RTCDataChannel | None = None
        # Attach a handler that will be called when the data channel opens.
        self.pc.on("datachannel", self._on_datachannel)
        logger.debug("AiortcHandler initialized, is_host=%s", is_host)
        if self.is_host:
            # Host creates the channel immediately.
            self.channel = self.pc.createDataChannel("control", ordered=True)
            self.channel.on("message", self._handle_message)
            self.channel.on("open", lambda: logger.info("Data channel opened (host)"))
            self.channel.on("close", lambda: logger.info("Data channel closed (host)"))
            self.channel.on("error", lambda e: logger.error(f"Data channel error (host): {e}"))
            logger.debug("Created data channel 'control' for host")

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
        self.channel.on("message", self._handle_message)
        self.channel.on("open", lambda: logger.info("Data channel opened (guest)"))
        self.channel.on("close", lambda: logger.info("Data channel closed (guest)"))
        self.channel.on("error", lambda e: logger.error(f"Data channel error (guest): {e}"))
        logger.debug("Received remote data channel")

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
        logger.debug("Sending payload over data channel: %s", payload)
        if self.channel is None:
            raise RuntimeError("Data channel not established yet")
        await self.channel.send(json.dumps(payload))

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

def run_async(coro: Any) -> Any:
    """Execute *coro* in a fresh event loop and return the result.
    This avoids "coroutine was never awaited" warnings when no loop is running.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

"""End of file
"""
