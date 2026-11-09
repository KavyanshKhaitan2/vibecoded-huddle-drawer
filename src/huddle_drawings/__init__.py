"""Draw on Slack huddle screen shares from Python.

No browser, no headless Chromium, no Slack app: joins a huddle over plain HTTP
plus one websocket, brings up a WebRTC audio leg so other clients accept it as
present, and sends the ``PeerMessage.draw`` protobufs that every Slack client
renders over the shared video.

Quick start -- invite this account to a huddle and draw on the first screen
share you see:

.. code-block:: python

    import asyncio
    from huddle_drawings import Config, HuddleSession, shape_segments

    async def main() -> None:
        config = Config.from_env()
        async with await HuddleSession.join(config, channel_id="C0123456789") as s:
            share = await s.wait_for_share()
            drawer = s.drawer(share, hue=200)
            await drawer.stroke(shape_segments("heart"))

    asyncio.run(main())

Joining puts a real user into a real call, visibly, and drawing acts on other
people's clients. Use a dedicated account in a huddle whose participants are in
on it.

Submodules with heavy dependencies (``audio`` needs aiortc, ``image_trace``
needs Pillow and numpy) are imported lazily, so importing this package stays
cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import (
    CHIME_DATA_MESSAGE_BURST,
    CHIME_DATA_MESSAGE_RATE,
    CLIENT_DATA_MESSAGE_BURST,
    CLIENT_DATA_MESSAGE_RATE,
    DATA_MESSAGE_HEADROOM,
    FADE_AFTER_END,
    FADE_AFTER_IDLE,
    FADE_DURATION,
    LINE_CLEAR_WAIT,
    MAX_DATA_MESSAGE_BYTES,
    TOPIC_AWARENESS,
    TOPIC_DATA_CHANNEL,
    TOPIC_TRANSCRIPT_CATCHUP,
    Config,
    ConfigError,
)
from .geometry import Point, Rect, Segment
from .shapes import SHAPES, shape_segments, svg_path_segments, text_in
from .types import (
    AttendeePresence,
    ChimeCredentials,
    DataMessage,
    HuddleError,
    HuddleJoin,
    SignalingError,
    SlackApiError,
    StreamSource,
)

if TYPE_CHECKING:
    from .autojoin import AutoJoiner, wait_for_invite
    from .draw import DrawClient
    from .realtime import HuddleInvite, RealtimeClient
    from .session import HuddleSession
    from .signaling import ChimeSignaling
    from .slack import SlackClient

__version__ = "0.1.0"

#: Names resolved on first access, mapped to the module that defines them.
_LAZY: dict[str, str] = {
    "AutoJoiner": "autojoin",
    "wait_for_invite": "autojoin",
    "DrawClient": "draw",
    "HuddleSession": "session",
    "ChimeSignaling": "signaling",
    "SlackClient": "slack",
    "RealtimeClient": "realtime",
    "HuddleInvite": "realtime",
    "AudioLeg": "audio",
    "force_srtp_profile": "audio",
    "trace_image": "image_trace",
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily-exported names above on first access."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CHIME_DATA_MESSAGE_BURST",
    "CHIME_DATA_MESSAGE_RATE",
    "CLIENT_DATA_MESSAGE_BURST",
    "CLIENT_DATA_MESSAGE_RATE",
    "DATA_MESSAGE_HEADROOM",
    "FADE_AFTER_END",
    "FADE_AFTER_IDLE",
    "FADE_DURATION",
    "LINE_CLEAR_WAIT",
    "MAX_DATA_MESSAGE_BYTES",
    "SHAPES",
    "TOPIC_AWARENESS",
    "TOPIC_DATA_CHANNEL",
    "TOPIC_TRANSCRIPT_CATCHUP",
    "AttendeePresence",
    "AudioLeg",
    "AutoJoiner",
    "ChimeCredentials",
    "ChimeSignaling",
    "Config",
    "ConfigError",
    "DataMessage",
    "DrawClient",
    "HuddleError",
    "HuddleInvite",
    "HuddleJoin",
    "HuddleSession",
    "Point",
    "RealtimeClient",
    "Rect",
    "Segment",
    "SignalingError",
    "SlackApiError",
    "SlackClient",
    "StreamSource",
    "__version__",
    "force_srtp_profile",
    "shape_segments",
    "svg_path_segments",
    "text_in",
    "trace_image",
    "wait_for_invite",
]
