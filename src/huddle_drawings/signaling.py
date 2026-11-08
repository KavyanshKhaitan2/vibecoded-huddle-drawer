"""The AWS Chime signaling websocket, reimplemented without the Chime SDK.

This is the control channel for a Slack huddle's media session. Everything here
follows ``docs/huddles-and-chime.md`` ("Joining without a browser", live-tested
2026-09-26) and ``docs/huddles-peer-messages.md`` ("Transport", "Throughput").

Shape of the protocol:

* Open ``<MediaPlacement.SignalingUrl>?X-Chime-Control-Protocol-Version=3
  &X-Amzn-Chime-Send-Close-On-Error=1`` with the websocket subprotocols
  ``['_aws_wt_session', <JoinToken>]``.
* Every websocket binary frame is **one byte ``0x05`` followed by a serialized
  ``SdkSignalFrame``** protobuf (the schema ships in ``amazon-chime-sdk-js`` as
  ``SignalingProtocol.js``; a copy is vendored in ``proto/``).
* ``JOIN`` → ``JOIN_ACK`` (carries TURN credentials), then ``SUBSCRIBE`` with an
  SDP offer → ``SUBSCRIBE_ACK`` with the answer, ``PING_PONG`` both ways,
  ``INDEX`` / ``AUDIO_STREAM_ID_INFO`` for state, ``DATA_MESSAGE`` for peer
  messages, and ``LEAVE`` → ``LEAVE_ACK`` on the way out.

Signaling alone is **not** enough to be a huddle participant: other clients only
treat us as present once Chime reports us in ``AUDIO_STREAM_ID_INFO``, which only
happens once real audio is flowing (see ``audio.py``). A signaling-only
connection can send data messages, but nobody renders them.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime, timezone
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.typing import Subprotocol

from .config import MAX_DATA_MESSAGE_BYTES
from .proto import signaling_protocol_pb2 as sp
from .ratelimit import TokenBucket
from .types import (
    CLOSE_CODE_MEANINGS,
    AttendeePresence,
    ChimeCredentials,
    DataMessage,
    SignalingError,
    StreamSource,
)

__all__ = [
    "ChimeSignaling",
    "SIGNAL_FRAME_PREFIX",
    "RECEIVED_FRAME_PREFIXES",
    "SIGNALING_SUBPROTOCOL",
    "bytes_to_frame",
    "frame_to_bytes",
    "signaling_url_for",
]

logger = logging.getLogger(__name__)

#: Prefix byte we put on every frame we send (the SDK's ``FRAME_TYPE_RTC``).
SIGNAL_FRAME_PREFIX: Final = 0x05

#: Prefix bytes Chime actually sends. The server uses **0x02** as well as
#: 0x05, and the payload after it is an ``SdkSignalFrame`` either way:
#: ``DefaultSignalingClient.stripFrameTypeRTC()`` in amazon-chime-sdk-js
#: accepts both, carrying the comment "TODO: change server frame type to send
#: 0x05". Observed live: a real huddle sends a steady stream of 0x02 frames,
#: and dropping them loses INDEX and AUDIO_STREAM_ID_INFO -- so the attendee
#: never learns it is present and its drawings are never shown.
RECEIVED_FRAME_PREFIXES: Final = frozenset({0x05, 0x02})

#: First websocket subprotocol; the second is the attendee's ``JoinToken``.
SIGNALING_SUBPROTOCOL: Final = "_aws_wt_session"

#: Query parameters the doc says to add to the ``SignalingUrl``.
_URL_PARAMS: Final = (
    ("X-Chime-Control-Protocol-Version", "3"),
    ("X-Amzn-Chime-Send-Close-On-Error", "1"),
)

#: ``SdkJoinFlags.HAS_STREAM_UPDATE``; the doc says JOIN sets this flag.
JOIN_FLAGS: Final = sp.SdkJoinFlags.Value("HAS_STREAM_UPDATE")

#: JOIN uses control protocol version 2 (the frame's own default, set anyway).
JOIN_PROTOCOL_VERSION: Final = 2

#: "Answer the server's PINGs with PONGs (and ping it every ~10 s)."
PING_INTERVAL: Final = 10.0

JOIN_ACK_TIMEOUT: Final = 20.0
SUBSCRIBE_ACK_TIMEOUT: Final = 30.0
#: Don't hang on the way out if the server never answers the LEAVE.
LEAVE_ACK_TIMEOUT: Final = 3.0

# The single audio send stream's descriptor. These are not guesses: they are
# the literal constants in `DefaultSignalingClient.subscribe()` in
# amazon-chime-sdk-js (src/signalingclient/DefaultSignalingClient.ts), which is
# the client Chime's server is built to accept. The `framerate` on an audio
# stream is odd but is genuinely what the SDK sends.
AUDIO_TRACK_LABEL: Final = "AmazonChimeExpressAudio"
_AUDIO_FRAMERATE: Final = 15
_AUDIO_MAX_BITRATE_KBPS: Final = 600
_AUDIO_AVG_BITRATE_BPS: Final = 400_000

_CallbackResult = None | Awaitable[None]
DataMessageCallback = Callable[[DataMessage], _CallbackResult]
IndexCallback = Callable[[list[StreamSource]], _CallbackResult]
AttendeesCallback = Callable[[list[AttendeePresence]], _CallbackResult]


# --------------------------------------------------------------------------- #
# frame codec
# --------------------------------------------------------------------------- #


def frame_to_bytes(frame: sp.SdkSignalFrame) -> bytes:
    """Serialize ``frame`` for the wire: ``0x05`` + the protobuf bytes."""
    return bytes((SIGNAL_FRAME_PREFIX,)) + frame.SerializeToString()


def bytes_to_frame(raw: bytes, *, strict: bool = False) -> sp.SdkSignalFrame:
    """Parse one websocket binary frame into an ``SdkSignalFrame``.

    The first byte is a frame type and is always stripped. Chime sends both
    ``0x05`` and ``0x02`` (see :data:`RECEIVED_FRAME_PREFIXES`), so both are
    accepted silently.

    An unrecognised prefix is *still decoded*, with a warning -- which is what
    the real SDK does, and is the safe choice: the server has already changed
    this byte once, and dropping frames on it loses ``INDEX`` and
    ``AUDIO_STREAM_ID_INFO``, which silently costs the attendee its presence
    and therefore its ability to draw. Pass ``strict=True`` to raise instead.

    Raises :class:`ValueError` for an empty or undecodable frame; callers log
    and ignore those rather than tearing the session down.
    """
    if not raw:
        raise ValueError("empty signaling frame")
    prefix = raw[0]
    if prefix not in RECEIVED_FRAME_PREFIXES:
        if strict:
            raise ValueError(f"unexpected signaling frame prefix 0x{prefix:02x}")
        logger.warning(
            "signaling frame has an unrecognised prefix 0x%02x; decoding it "
            "anyway (the SDK does the same)",
            prefix,
        )
    frame = sp.SdkSignalFrame()
    try:
        frame.ParseFromString(raw[1:])
    except Exception as exc:  # protobuf raises DecodeError and friends
        raise ValueError(f"undecodable SdkSignalFrame: {exc}") from exc
    return frame


def signaling_url_for(signaling_url: str) -> str:
    """Add the two required query parameters to a ``MediaPlacement.SignalingUrl``.

    Any query string already on the URL is preserved (Chime's signaling URLs
    have carried per-session parameters), and a parameter that is already there
    is left alone.
    """
    parts = urlsplit(signaling_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    present = {key for key, _ in query}
    query.extend((key, value) for key, value in _URL_PARAMS if key not in present)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _rewrite_origin_line(sdp: str) -> str:
    """Rewrite the SDP's ``o=-`` origin line to ``o=mozilla-chrome``.

    ``docs/huddles-and-chime.md``: "send ``SUBSCRIBE`` with the SDP offer (the
    SDK rewrites ``o=-`` to ``o=mozilla-chrome``)". This mirrors
    ``SDP.withUnifiedPlanFormat()`` in ``amazon-chime-sdk-js``, which replaces
    the first ``o=-`` unless the SDP already mentions ``mozilla``; Chime's
    signaling server uses it to decide the SDP dialect it answers in.
    """
    if "mozilla" in sdp:
        return sdp
    return sdp.replace("o=-", "o=mozilla-chrome", 1)


def _client_details() -> sp.SdkClientDetails:
    """Plausible ``clientDetails`` for JOIN.

    The doc only requires "any plausible clientDetails"; these mimic the
    browser-hosted Chime SDK that Slack itself ships.
    """
    offset = datetime.now(timezone.utc).astimezone().strftime("%z") or "+0000"
    return sp.SdkClientDetails(
        app_name="huddle-drawings",
        app_version="0.1.0",
        device_model="Chrome",
        device_make="Google",
        platform_name="chrome",
        platform_version="131.0.0",
        client_source="amazon-chime-sdk-js",
        chime_sdk_version="3.32.0",
        client_utc_offset=f"{offset[:3]}:{offset[3:]}",
        os_name="linux",
        os_version="6",
    )


def _stream_source(descriptor: sp.SdkStreamDescriptor) -> StreamSource:
    """Convert one ``SdkStreamDescriptor`` to a :class:`StreamSource`."""
    media_type = (
        sp.SdkStreamMediaType.Name(descriptor.media_type)
        if descriptor.HasField("media_type")
        else None
    )
    return StreamSource(
        stream_id=descriptor.stream_id,
        attendee_id=descriptor.attendee_id,
        external_user_id=descriptor.external_user_id or None,
        media_type=media_type,
        width=descriptor.width if descriptor.HasField("width") else None,
        height=descriptor.height if descriptor.HasField("height") else None,
        group_id=descriptor.group_id if descriptor.HasField("group_id") else None,
        max_bitrate_kbps=(
            descriptor.max_bitrate_kbps
            if descriptor.HasField("max_bitrate_kbps")
            else None
        ),
    )


def _attendee_presence(info: sp.SdkAudioStreamIdInfo) -> AttendeePresence:
    return AttendeePresence(
        audio_stream_id=info.audio_stream_id,
        attendee_id=info.attendee_id,
        external_user_id=info.external_user_id or None,
        muted=info.muted,
        dropped=info.dropped,
    )


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #


class ChimeSignaling:
    """One Chime signaling websocket for one attendee of one huddle.

    Usage::

        sig = ChimeSignaling(join.chime)
        await sig.connect()
        answer = await sig.subscribe(sdp_offer=offer, audio_host=host)
        ...
        await sig.leave()
        await sig.close()

    All public coroutines raise :class:`SignalingError` once the websocket has
    gone away; the error carries the close code and, for the codes seen in the
    wild (1006 / 4403 / 4410), an explanation from
    :data:`huddle_drawings.types.CLOSE_CODE_MEANINGS`.
    """

    def __init__(
        self,
        creds: ChimeCredentials,
        *,
        rate_limiter: TokenBucket | None = None,
    ) -> None:
        self._creds = creds
        #: Our own Chime attendee id -- the ``itemId``/``fromPeerIdString`` value.
        self.attendee_id: str = creds.attendee_id
        #: Latest ``INDEX``, keyed by ``stream_id``. Replaced wholesale per frame.
        self.sources: dict[int, StreamSource] = {}
        #: Latest ``AUDIO_STREAM_ID_INFO``, keyed by attendee id. Incremental.
        self.attendees: dict[str, AttendeePresence] = {}

        self._rate_limiter = rate_limiter if rate_limiter is not None else TokenBucket()
        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._turn_credentials: sp.SdkTurnCredentials | None = None

        self._waiters: dict[int, list[asyncio.Future[sp.SdkSignalFrame]]] = {}
        self._state_waiters: list[asyncio.Future[None]] = []
        self._send_lock = asyncio.Lock()

        self._data_callbacks: list[DataMessageCallback] = []
        self._index_callbacks: list[IndexCallback] = []
        self._attendee_callbacks: list[AttendeesCallback] = []

        self._failure: SignalingError | None = None
        self._closing = False
        self._ping_id = 0
        # The SDK generates a random audio session id per JOIN.
        self._audio_session_id = secrets.randbits(32)

    # -- plumbing ---------------------------------------------------------- #

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return (
            f"<ChimeSignaling {state} attendee={self.attendee_id!r} "
            f"sources={len(self.sources)} attendees={len(self.attendees)}>"
        )

    @property
    def signaling_url(self) -> str:
        """The websocket URL we connect to, parameters included."""
        return signaling_url_for(self._creds.media_placement.signaling_url)

    @property
    def is_connected(self) -> bool:
        """True while the websocket is open and no failure has been recorded."""
        return self._ws is not None and self._failure is None and not self._closing

    @property
    def turn_credentials(self) -> sp.SdkTurnCredentials | None:
        """TURN credentials from ``JOIN_ACK``, if the server sent any."""
        return self._turn_credentials

    async def __aenter__(self) -> ChimeSignaling:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- connect / JOIN ---------------------------------------------------- #

    async def connect(self) -> None:
        """Open the websocket, do the JOIN/JOIN_ACK handshake, start the reader.

        The reader task is started *before* JOIN is sent so the ``JOIN_ACK``
        cannot be missed.
        """
        if self._ws is not None:
            raise SignalingError("already connected to Chime signaling")

        url = self.signaling_url
        logger.info(
            "opening Chime signaling websocket for meeting %s attendee %s",
            self._creds.meeting_id,
            self.attendee_id,
        )
        try:
            self._ws = await connect(
                url,
                # The JoinToken rides in the subprotocol list; there is no
                # Authorization header and no token in the query string.
                subprotocols=[
                    Subprotocol(SIGNALING_SUBPROTOCOL),
                    Subprotocol(self._creds.join_token),
                ],
                # Chime keeps the connection alive with its own PING_PONG
                # frames, so RFC6455 ping/pong is off: an unanswered
                # websocket-level ping would kill a healthy session.
                ping_interval=None,
                open_timeout=20.0,
                close_timeout=5.0,
                # Room for a 200-payload data-message burst (see the doc's
                # throughput section) without stalling the socket.
                max_queue=512,
            )
        except (OSError, WebSocketException) as exc:
            raise SignalingError(
                f"could not open the Chime signaling websocket: {exc}"
            ) from exc

        self._failure = None
        self._closing = False
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="chime-signaling-reader"
        )

        try:
            join = self._new_frame(sp.SdkSignalFrame.JOIN)
            join.join.protocol_version = JOIN_PROTOCOL_VERSION
            join.join.flags = JOIN_FLAGS
            join.join.max_num_of_videos = 1
            join.join.client_details.CopyFrom(_client_details())
            # DefaultSignalingClient.join() sets a random audio_session_id on
            # every JOIN. Verified in amazon-chime-sdk-js
            # src/signalingclient/DefaultSignalingClient.ts.
            join.join.audio_session_id = self._audio_session_id
            ack = await self._request(join, sp.SdkSignalFrame.JOIN_ACK, JOIN_ACK_TIMEOUT)
        except BaseException:
            await self.close()
            raise

        if ack.joinack.HasField("turn_credentials"):
            self._turn_credentials = ack.joinack.turn_credentials
        logger.info(
            "joined Chime meeting %s (turn_credentials=%s, video_subscription_limit=%s)",
            self._creds.meeting_id,
            "yes" if self._turn_credentials is not None else "no",
            ack.joinack.video_subscription_limit
            if ack.joinack.HasField("video_subscription_limit")
            else "-",
        )

        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="chime-signaling-keepalive"
        )

    # -- SUBSCRIBE --------------------------------------------------------- #

    async def subscribe(
        self,
        *,
        sdp_offer: str,
        audio_host: str,
        receive_stream_ids: Sequence[int] = (0,),
        audio_muted: bool = True,
        send_audio_stream_id: int = 1,
    ) -> str:
        """Send ``SUBSCRIBE``, wait for ``SUBSCRIBE_ACK``, return the SDP answer.

        ``sdp_offer`` must contain an audio section *and* a ``recvonly`` video
        section: without a video section the ack comes back with "failed to
        initialize video session" (``docs/huddles-and-chime.md``). Put a screen
        share's ``stream_id`` (from :attr:`sources`) in ``receive_stream_ids`` to
        receive its video; ``0`` alone is enough for audio only.
        """
        frame = self._new_frame(sp.SdkSignalFrame.SUBSCRIBE)
        sub = frame.sub
        sub.duplex = sp.SdkStreamServiceType.Value("RX")
        sub.sdp_offer = _rewrite_origin_line(sdp_offer)
        sub.audio_host = audio_host
        sub.audio_muted = audio_muted
        sub.receive_stream_ids.extend(receive_stream_ids)
        # Exactly one audio send stream, as the doc specifies ("one audio
        # sendStreams entry"). Every field below is copied from the real SDK's
        # `DefaultSignalingClient.subscribe()`
        # (amazon-chime-sdk-js src/signalingclient/DefaultSignalingClient.ts),
        # which builds this descriptor with exactly these constants -- including
        # a framerate on an *audio* stream, which looks wrong but is what the
        # SDK sends, so we match it rather than second-guess the server.
        send_stream = sub.send_streams.add()
        send_stream.media_type = sp.SdkStreamMediaType.Value("AUDIO")
        send_stream.attendee_id = self.attendee_id
        send_stream.stream_id = send_audio_stream_id
        send_stream.group_id = send_audio_stream_id
        send_stream.track_label = AUDIO_TRACK_LABEL
        send_stream.framerate = _AUDIO_FRAMERATE
        send_stream.max_bitrate_kbps = _AUDIO_MAX_BITRATE_KBPS
        send_stream.avg_bitrate_bps = _AUDIO_AVG_BITRATE_BPS

        logger.debug(
            "sending SUBSCRIBE (receive_stream_ids=%s, audio_muted=%s, audio_host=%s)",
            list(receive_stream_ids),
            audio_muted,
            audio_host,
        )
        ack = await self._request(
            frame, sp.SdkSignalFrame.SUBSCRIBE_ACK, SUBSCRIBE_ACK_TIMEOUT
        )

        if ack.HasField("error") and (
            ack.error.description or ack.error.HasField("status")
        ):
            description = ack.error.description or "(no description)"
            hint = ""
            if "video session" in description:
                hint = (
                    " -- hint: Chime says this when the SDP offer has no video "
                    "section; add a recvonly video transceiver to the offer"
                )
            raise SignalingError(
                f"SUBSCRIBE_ACK failed (status {ack.error.status}): {description}{hint}"
            )
        if not ack.suback.sdp_answer:
            raise SignalingError("SUBSCRIBE_ACK carried no sdp_answer")
        return ack.suback.sdp_answer

    # -- data messages ----------------------------------------------------- #

    async def send_data_message(
        self,
        topic: str,
        data: bytes,
        *,
        lifetime_ms: int = 0,
        wait: bool = True,
    ) -> bool:
        """Send one ``DATA_MESSAGE`` payload, rate-limited.

        Chime's limit is a **per-meeting token bucket of ~200 payloads refilling
        at ~150/s, shared by every sender in the meeting**. Excess is **dropped
        silently** -- receivers never see a ``throttled`` flag and the sender
        gets no feedback at all -- and sustained overload (800+/s) disconnects
        the sender with close code **1006**
        (``docs/huddles-peer-messages.md#throughput``). So we police ourselves
        with a :class:`~huddle_drawings.ratelimit.TokenBucket`.

        ``wait=True`` waits for a token; ``wait=False`` returns ``False``
        immediately instead of blocking (prefer that for anything that is better
        skipped than delayed, e.g. cursor updates).

        Returns whether the payload was handed to the websocket. Raises
        :class:`ValueError` for a payload over
        :data:`~huddle_drawings.config.MAX_DATA_MESSAGE_BYTES`, and
        :class:`SignalingError` if the channel is gone.
        """
        if len(data) > MAX_DATA_MESSAGE_BYTES:
            raise ValueError(
                f"data message payload is {len(data)} bytes, over Chime's "
                f"{MAX_DATA_MESSAGE_BYTES}-byte limit (topic {topic!r})"
            )
        # Fail before spending a token (or waiting for one) on a dead channel.
        # The Slack client does the same with `signalingClient.ready()`.
        if self._failure is not None:
            raise self._failure
        if self._ws is None:
            raise SignalingError("not connected to Chime signaling")

        if wait:
            waited = await self._rate_limiter.acquire()
            if waited > 0.01:
                logger.debug("rate limiter delayed a %r message by %.3fs", topic, waited)
        elif not self._rate_limiter.try_acquire():
            logger.debug("dropping a %r data message: rate limiter is empty", topic)
            return False

        frame = self._new_frame(sp.SdkSignalFrame.DATA_MESSAGE)
        payload = frame.data_message.messages.add()
        payload.topic = topic
        payload.data = data
        payload.lifetime_ms = lifetime_ms
        await self._send_frame(frame)
        return True

    # -- callbacks --------------------------------------------------------- #

    def on_data_message(self, cb: DataMessageCallback) -> Callable[[], None]:
        """Register a ``DATA_MESSAGE`` callback; returns an unsubscribe callable.

        The callback may be sync or async. Callbacks are invoked in registration
        order and awaited in turn, so receive order is preserved -- keep them
        quick. One raising is logged and does not stop the others or the reader.
        """
        return self._register(self._data_callbacks, cb)

    def on_index(self, cb: IndexCallback) -> Callable[[], None]:
        """Register an ``INDEX`` callback (full source list); returns unsubscribe."""
        return self._register(self._index_callbacks, cb)

    def on_attendees(self, cb: AttendeesCallback) -> Callable[[], None]:
        """Register an ``AUDIO_STREAM_ID_INFO`` callback; returns unsubscribe.

        The list holds only the entries in that frame -- these updates are
        incremental, and an entry with ``dropped=True`` means that attendee is
        gone. :attr:`attendees` holds the merged view.
        """
        return self._register(self._attendee_callbacks, cb)

    @staticmethod
    def _register(
        registry: list[Any], cb: Any
    ) -> Callable[[], None]:
        registry.append(cb)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                registry.remove(cb)

        return unsubscribe

    # -- waiting for state ------------------------------------------------- #

    async def wait_for_attendee(
        self, attendee_id: str, timeout: float = 15.0
    ) -> AttendeePresence:
        """Wait until Chime reports ``attendee_id`` in ``AUDIO_STREAM_ID_INFO``.

        Returns immediately if that attendee is already known. An attendee only
        shows up here once its audio connection is up, which is also when Slack
        clients start accepting its draw messages.

        Raises :class:`TimeoutError` if it never appears, or
        :class:`SignalingError` if the channel fails while waiting.
        """

        def find() -> AttendeePresence | None:
            presence = self.attendees.get(attendee_id)
            return presence if presence is not None and not presence.dropped else None

        return await self._wait_for_state(
            find, timeout, f"attendee {attendee_id!r} to join"
        )

    async def wait_for_content_share(
        self, timeout: float | None = None
    ) -> StreamSource:
        """Wait for a screen share's video source to appear in ``INDEX``.

        Returns the first source whose ``attendee_id`` ends in ``#content`` and
        whose media type is video; use its
        :attr:`~huddle_drawings.types.StreamSource.sharer_attendee_id` as the
        draw messages' ``itemId``. A share's attendee id changes every time the
        share restarts, so long-running drawers should keep following
        :meth:`on_index` rather than hold on to this value.

        ``timeout=None`` waits forever. Raises :class:`TimeoutError` on timeout.
        """

        def find() -> StreamSource | None:
            for source in self.sources.values():
                if source.is_content_share and source.media_type == "VIDEO":
                    return source
            return None

        return await self._wait_for_state(find, timeout, "a content share")

    async def _wait_for_state[T](
        self,
        find: Callable[[], T | None],
        timeout: float | None,
        what: str,
    ) -> T:
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while True:
            # Register before checking, so an update between the check and the
            # await cannot be missed.
            waiter: asyncio.Future[None] = loop.create_future()
            self._state_waiters.append(waiter)
            try:
                if self._failure is not None:
                    raise self._failure
                found = find()
                if found is not None:
                    return found
                remaining = None if deadline is None else deadline - loop.time()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out after {timeout}s waiting for {what}")
                try:
                    await asyncio.wait_for(waiter, remaining)
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"timed out after {timeout}s waiting for {what}"
                    ) from exc
            finally:
                with contextlib.suppress(ValueError):
                    self._state_waiters.remove(waiter)

    # -- leave / close ----------------------------------------------------- #

    async def leave(self) -> None:
        """Send ``LEAVE`` and wait briefly for ``LEAVE_ACK``.

        Leaving cleanly matters: a connection that does not leave makes the next
        join of the same account fail with close code 4403 "attendee
        unavailable". Never raises -- a failure here is logged, since the caller
        is on its way out anyway.
        """
        if self._ws is None or self._failure is not None:
            logger.debug("skipping LEAVE: no live signaling connection")
            return
        self._closing = True
        try:
            await self._request(
                self._new_frame(sp.SdkSignalFrame.LEAVE),
                sp.SdkSignalFrame.LEAVE_ACK,
                LEAVE_ACK_TIMEOUT,
            )
            logger.info("left Chime meeting %s", self._creds.meeting_id)
        except (SignalingError, TimeoutError, OSError, WebSocketException) as exc:
            logger.warning("LEAVE was not acknowledged: %s", exc)

    async def close(self) -> None:
        """Cancel the background tasks and close the websocket. Idempotent.

        Safe to call after an error, after :meth:`leave`, before
        :meth:`connect`, and from inside a callback.
        """
        self._closing = True
        ws, self._ws = self._ws, None
        tasks = (self._keepalive_task, self._reader_task)
        self._keepalive_task = self._reader_task = None

        current = asyncio.current_task()
        for task in tasks:
            if task is None or task is current:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
            logger.debug("Chime signaling websocket closed")

        self._fail(
            SignalingError("the Chime signaling connection was closed locally"),
            quiet=True,
        )

    # -- internals: framing and request/response --------------------------- #

    @staticmethod
    def _new_frame(frame_type: int) -> sp.SdkSignalFrame:
        frame = sp.SdkSignalFrame()
        frame.type = frame_type
        # timestamp_ms is required and is set again in _send_frame, as close to
        # the write as possible.
        frame.timestamp_ms = int(time.time() * 1000)
        return frame

    async def _send_frame(self, frame: sp.SdkSignalFrame) -> None:
        if self._failure is not None:
            raise self._failure
        ws = self._ws
        if ws is None:
            raise SignalingError("not connected to Chime signaling")
        frame.timestamp_ms = int(time.time() * 1000)
        raw = frame_to_bytes(frame)
        try:
            async with self._send_lock:
                await ws.send(raw)
        except ConnectionClosed as exc:
            error = self._close_error(ws)
            self._fail(error)
            raise error from exc
        except (OSError, WebSocketException) as exc:
            error = SignalingError(f"failed to send a signaling frame: {exc}")
            self._fail(error)
            raise error from exc

    def _expect(self, frame_type: int) -> asyncio.Future[sp.SdkSignalFrame]:
        """Register a one-shot waiter for the next frame of ``frame_type``."""
        future: asyncio.Future[sp.SdkSignalFrame] = (
            asyncio.get_running_loop().create_future()
        )
        if self._failure is not None:
            future.set_exception(self._failure)
        else:
            self._waiters.setdefault(frame_type, []).append(future)
        return future

    async def _request(
        self,
        frame: sp.SdkSignalFrame,
        reply_type: int,
        timeout: float,
    ) -> sp.SdkSignalFrame:
        """Send ``frame`` and wait for the matching reply frame type."""
        future = self._expect(reply_type)
        try:
            await self._send_frame(frame)
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as exc:
            raise SignalingError(
                f"timed out after {timeout}s waiting for "
                f"{sp.SdkSignalFrame.Type.Name(reply_type)}"
            ) from exc
        finally:
            waiters = self._waiters.get(reply_type)
            if waiters is not None:
                with contextlib.suppress(ValueError):
                    waiters.remove(future)

    def _close_error(self, ws: ClientConnection | None = None) -> SignalingError:
        """Build the :class:`SignalingError` for a closed websocket."""
        ws = ws if ws is not None else self._ws
        code = ws.close_code if ws is not None else None
        reason = (ws.close_reason or "") if ws is not None else ""
        message = f"the Chime signaling websocket closed (code {code}"
        if reason:
            message += f", {reason!r}"
        message += ")"
        meaning = CLOSE_CODE_MEANINGS.get(code) if code is not None else None
        if meaning:
            message += f": {meaning}"
        return SignalingError(message, code=code)

    def _fail(self, error: SignalingError, *, quiet: bool = False) -> None:
        """Record a terminal failure and hand it to everyone who is waiting.

        A reader or keepalive task that dies must not leave waiters hanging, so
        every pending ack waiter and state waiter gets the error.
        """
        first = self._failure is None
        if first:
            self._failure = error
            if not quiet:
                if self._closing:
                    logger.debug("signaling ended while closing: %s", error)
                else:
                    logger.warning("Chime signaling failed: %s", error)

        waiters = self._waiters
        self._waiters = {}
        for futures in waiters.values():
            for future in futures:
                if not future.done():
                    future.set_exception(self._failure or error)
        for state_waiter in list(self._state_waiters):
            if not state_waiter.done():
                state_waiter.set_exception(self._failure or error)
        self._state_waiters.clear()

    def _notify_state(self) -> None:
        """Wake :meth:`wait_for_attendee` / :meth:`wait_for_content_share`."""
        waiters, self._state_waiters = self._state_waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    # -- internals: the reader --------------------------------------------- #

    async def _read_loop(self) -> None:
        ws = self._ws
        assert ws is not None
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    logger.warning(
                        "ignoring an unexpected text frame from Chime signaling "
                        "(%d chars)",
                        len(raw),
                    )
                    continue
                try:
                    frame = bytes_to_frame(raw)
                except ValueError as exc:
                    logger.warning("ignoring a signaling frame: %s", exc)
                    continue
                await self._dispatch(frame)
        except asyncio.CancelledError:
            self._fail(
                SignalingError("the Chime signaling reader was cancelled"), quiet=True
            )
            raise
        except ConnectionClosed:
            self._fail(self._close_error(ws))
        except Exception as exc:  # never die silently: strand nobody
            logger.exception("the Chime signaling reader crashed")
            self._fail(SignalingError(f"the Chime signaling reader crashed: {exc}"))
        else:
            # Iteration ends without raising on a normal close (1000/1001).
            self._fail(self._close_error(ws))

    async def _keepalive_loop(self) -> None:
        """PING the server every ~10s, as the doc requires."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                self._ping_id = (self._ping_id % 0xFFFFFFFF) + 1
                frame = self._new_frame(sp.SdkSignalFrame.PING_PONG)
                frame.ping_pong.type = sp.SdkPingPongType.Value("PING")
                frame.ping_pong.ping_id = self._ping_id
                await self._send_frame(frame)
                logger.debug("sent signaling PING %d", self._ping_id)
        except asyncio.CancelledError:
            raise
        except SignalingError:
            # _send_frame already recorded the failure and woke the waiters.
            pass
        except Exception as exc:
            logger.exception("the Chime signaling keepalive crashed")
            self._fail(SignalingError(f"the Chime signaling keepalive crashed: {exc}"))

    async def _dispatch(self, frame: sp.SdkSignalFrame) -> None:
        frame_type = frame.type
        try:
            name = sp.SdkSignalFrame.Type.Name(frame_type)
        except ValueError:
            name = f"UNKNOWN({frame_type})"

        if frame.HasField("error") and (
            frame.error.description or frame.error.HasField("status")
        ):
            logger.warning(
                "%s frame carried an error: status=%s %s",
                name,
                frame.error.status,
                frame.error.description,
            )

        if frame_type == sp.SdkSignalFrame.PING_PONG:
            await self._handle_ping_pong(frame)
        elif frame_type == sp.SdkSignalFrame.INDEX:
            await self._handle_index(frame)
        elif frame_type == sp.SdkSignalFrame.AUDIO_STREAM_ID_INFO:
            await self._handle_attendees(frame)
        elif frame_type == sp.SdkSignalFrame.DATA_MESSAGE:
            await self._handle_data_message(frame)
        elif frame_type == sp.SdkSignalFrame.NOTIFICATION:
            logger.warning(
                "Chime notification (%s): %s",
                sp.SdkNotificationFrame.NotificationLevel.Name(
                    frame.notification.level
                ),
                frame.notification.message,
            )
        else:
            logger.debug("received %s frame", name)

        for future in self._waiters.pop(frame_type, []):
            if not future.done():
                future.set_result(frame)

    async def _handle_ping_pong(self, frame: sp.SdkSignalFrame) -> None:
        ping_pong = frame.ping_pong
        if ping_pong.type == sp.SdkPingPongType.Value("PING"):
            # Answer with the *same* ping_id or Chime considers us unresponsive.
            pong = self._new_frame(sp.SdkSignalFrame.PING_PONG)
            pong.ping_pong.type = sp.SdkPingPongType.Value("PONG")
            pong.ping_pong.ping_id = ping_pong.ping_id
            await self._send_frame(pong)
            logger.debug("answered signaling PING %d", ping_pong.ping_id)
        else:
            logger.debug("received signaling PONG %d", ping_pong.ping_id)

    async def _handle_index(self, frame: sp.SdkSignalFrame) -> None:
        # INDEX is a full replacement, not a delta: a screen share that restarts
        # comes back under a new attendee id (``<sharer>#content``) and a new
        # stream id, and the old entry simply stops being listed.
        sources = [_stream_source(d) for d in frame.index.sources]
        self.sources = {source.stream_id: source for source in sources}
        logger.debug(
            "INDEX: %d source(s) %s",
            len(sources),
            [(s.stream_id, s.attendee_id, s.media_type) for s in sources],
        )
        self._notify_state()
        await self._emit(self._index_callbacks, sources, "on_index")

    async def _handle_attendees(self, frame: sp.SdkSignalFrame) -> None:
        entries = [_attendee_presence(i) for i in frame.audio_stream_id_info.streams]
        for entry in entries:
            if entry.dropped:
                attendee_id = entry.attendee_id or self._attendee_for_stream(
                    entry.audio_stream_id
                )
                if attendee_id:
                    self.attendees.pop(attendee_id, None)
                    logger.debug("attendee %s dropped", attendee_id)
                continue
            if not entry.attendee_id:
                logger.debug(
                    "ignoring an AUDIO_STREAM_ID_INFO entry with no attendee_id: %s",
                    entry,
                )
                continue
            self.attendees[entry.attendee_id] = entry
        self._notify_state()
        await self._emit(self._attendee_callbacks, entries, "on_attendees")

    def _attendee_for_stream(self, audio_stream_id: int) -> str | None:
        """Reverse-lookup for ``dropped`` entries that omit the attendee id."""
        for attendee_id, presence in self.attendees.items():
            if presence.audio_stream_id == audio_stream_id:
                return attendee_id
        return None

    async def _handle_data_message(self, frame: sp.SdkSignalFrame) -> None:
        for payload in frame.data_message.messages:
            message = DataMessage(
                topic=payload.topic,
                data=bytes(payload.data),
                sender_attendee_id=payload.sender_attendee_id,
                sender_external_user_id=payload.sender_external_user_id or None,
                lifetime_ms=payload.lifetime_ms,
                ingest_time_ns=(
                    payload.ingest_time_ns
                    if payload.HasField("ingest_time_ns")
                    else None
                ),
            )
            await self._emit(self._data_callbacks, message, "on_data_message")

    async def _emit(
        self, callbacks: Iterable[Any], payload: Any, kind: str
    ) -> None:
        """Call every callback, awaiting async ones; log and swallow failures."""
        for cb in list(callbacks):
            try:
                result = cb(payload)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("a %s callback raised; continuing", kind)
