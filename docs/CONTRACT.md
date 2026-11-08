# Internal module contract

This file pins down the interfaces between modules so they can be built
independently. **Do not change a signature listed here** — other modules are
written against it. If something here is genuinely wrong, say so in your report
instead of silently diverging.

Reference docs (read these, they are the spec):

- `docs/huddles-peer-messages.md` — what travels inside a huddle's Chime
  session: the `PeerMessage` protobuf, the drawing rules, fade timings, the
  per-meeting rate limit.
- `docs/huddles-and-chime.md` — how to join: `rooms.join`, the Chime signaling
  websocket, the required audio leg, receiving video, leaving.

Already written and working (do not rewrite):

- `src/huddle_drawings/config.py` — `Config.from_env()`, protocol constants
  (`TOPIC_DATA_CHANNEL`, `MAX_DATA_MESSAGE_BYTES`, fade timings…).
- `src/huddle_drawings/types.py` — all shared dataclasses and exceptions.
- `src/huddle_drawings/ratelimit.py` — `TokenBucket` (async `acquire`,
  sync `try_acquire`).
- `src/huddle_drawings/proto/{signaling_protocol,peer_message}_pb2.py` —
  generated; import as
  `from huddle_drawings.proto import signaling_protocol_pb2 as sp`.
  Regenerate with `make proto` if a `.proto` changes.

Style: Python 3.13+, `from __future__ import annotations`, asyncio, full type
hints, docstrings that cite the doc for any non-obvious protocol rule. No
`print()` — use `logging.getLogger(__name__)`.

---

## `slack.py` — Slack Web API (browser-session identity)

```python
class SlackClient:
    def __init__(self, config: Config, *, client: httpx.AsyncClient | None = None) -> None
    async def __aenter__(self) -> SlackClient
    async def __aexit__(self, *exc: object) -> None
    async def close(self) -> None

    async def call(self, method: str, params: dict[str, Any] | None = None, *,
                   reason: str | None = None) -> dict[str, Any]
        """POST to https://<host>/api/<method>. Raises SlackApiError on ok:false."""

    async def auth_test(self) -> dict[str, Any]
    async def rooms_join(self, channel_id: str, *, regions: str | None = None) -> HuddleJoin
    async def rooms_info(self, room_id: str) -> dict[str, Any]   # screenhero.rooms.info
    async def rooms_leave(self, room_id: str) -> dict[str, Any]  # may return feature_not_enabled
    async def rooms_invite_response(self, *, channel_id: str, room_id: str,
                                    response: str = "decline") -> dict[str, Any]
```

## `signaling.py` — Chime signaling websocket

```python
class ChimeSignaling:
    def __init__(self, creds: ChimeCredentials, *,
                 rate_limiter: TokenBucket | None = None) -> None

    attendee_id: str
    sources: dict[int, StreamSource]       # latest INDEX
    attendees: dict[str, AttendeePresence]  # latest AUDIO_STREAM_ID_INFO

    async def connect(self) -> None          # websocket + JOIN/JOIN_ACK, starts reader
    async def subscribe(self, *, sdp_offer: str, audio_host: str,
                        receive_stream_ids: Sequence[int] = (0,),
                        audio_muted: bool = True,
                        send_audio_stream_id: int = 1) -> str
        """Send SUBSCRIBE, wait for SUBSCRIBE_ACK, return the SDP answer."""
    async def send_data_message(self, topic: str, data: bytes, *,
                                lifetime_ms: int = 0, wait: bool = True) -> bool
        """Rate-limited. wait=False drops instead of blocking; returns whether sent."""
    async def leave(self) -> None            # LEAVE / LEAVE_ACK
    async def close(self) -> None

    def on_data_message(self, cb: Callable[[DataMessage], None]) -> Callable[[], None]
    def on_index(self, cb: Callable[[list[StreamSource]], None]) -> Callable[[], None]
    def on_attendees(self, cb: Callable[[list[AttendeePresence]], None]) -> Callable[[], None]
        """Each returns an unsubscribe callable. Callbacks may be async."""

    async def wait_for_attendee(self, attendee_id: str, timeout: float = 15.0) -> AttendeePresence
    async def wait_for_content_share(self, timeout: float | None = None) -> StreamSource
    @property
    def turn_credentials(self) -> SdkTurnCredentials | None
```

Wire format: each websocket binary frame is one byte `0x05` followed by a
serialized `SdkSignalFrame`.

## `audio.py` — the required media leg (aiortc)

```python
def force_srtp_profile() -> None
    """Restrict aiortc to SRTP_AES128_CM_SHA1_80 — mandatory, see the doc."""

class AudioLeg:
    def __init__(self, *, muted: bool = True) -> None
    async def create_offer(self) -> str
    async def accept_answer(self, sdp_answer: str) -> None
    async def close(self) -> None
    @property
    def is_running(self) -> bool
```

## `peer_messages.py` — PeerMessage builders/parsers

Builders return `bytes` ready for `send_data_message`; all set
`fromPeerIdString`. One builder per field in the schema, plus
`parse_peer_message(data: bytes) -> PeerMessage` and a
`describe_peer_message(msg) -> str` for logs.

## `geometry.py` — pure geometry, no I/O

Normalized 0..1 coordinates over the shared video. Catmull-Rom → cubic Bézier
fitting, polyline simplification (Ramer–Douglas–Peucker), the "hairpin" fill
from the doc, plus shape helpers (line, rect, circle, polyline, text).

## `draw.py` — the drawing client

```python
class DrawClient:
    def __init__(self, sender: DataMessageSender, *, attendee_id: str,
                 item_id: str, hue: int | None = None) -> None
    async def set_hue(self, hue: int) -> None
    async def stroke(self, segments: Sequence[Segment], *, end: bool = True) -> None
    async def draw_path(self, points: Sequence[Point], *, end: bool = True) -> None
    async def move_cursor(self, x: float, y: float) -> None
    async def hide_cursor(self) -> None
    async def end_stroke(self, *, twice: bool = True) -> None
    async def keep_alive(self) -> None
    def start_keep_alive(self) -> None   # background task, re-sends inside 5s
    async def stop_keep_alive(self) -> None
    async def wait_for_clear(self) -> None  # ~8.5s after END
```

`DataMessageSender` is a `Protocol` with
`async def send_data_message(self, topic: str, data: bytes, *, lifetime_ms: int = 0, wait: bool = True) -> bool`.
`ChimeSignaling` satisfies it; tests use a fake.
