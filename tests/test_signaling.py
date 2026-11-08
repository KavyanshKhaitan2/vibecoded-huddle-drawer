"""Tests for ``signaling.py`` against a local websocket server that speaks the
real Chime signaling wire format (``0x05`` + ``SdkSignalFrame``).

No network: the fake server listens on 127.0.0.1 on an ephemeral port.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from huddle_drawings import signaling
from huddle_drawings.proto import signaling_protocol_pb2 as sp
from huddle_drawings.ratelimit import TokenBucket
from huddle_drawings.signaling import (
    RECEIVED_FRAME_PREFIXES,
    SIGNALING_SUBPROTOCOL,
    ChimeSignaling,
    bytes_to_frame,
    frame_to_bytes,
    signaling_url_for,
)
from huddle_drawings.types import (
    AttendeePresence,
    ChimeCredentials,
    DataMessage,
    MediaPlacement,
    SignalingError,
    StreamSource,
)

ATTENDEE = "attendee-self"
JOIN_TOKEN = "join-token-abc123"
OFFER = (
    "v=0\r\n"
    "o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n"
    "s=-\r\n"
    "t=0 0\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
    "m=video 9 UDP/TLS/RTP/SAVPF 102\r\n"
)
ANSWER = "v=0\r\no=mozilla-chrome 1 2 IN IP4 127.0.0.1\r\ns=-\r\n"


# --------------------------------------------------------------------------- #
# the fake Chime signaling server
# --------------------------------------------------------------------------- #


class FakeChimeServer:
    """A local websocket server that answers the frames ``signaling.py`` sends."""

    def __init__(self) -> None:
        self.server: Any = None
        self.port = 0
        self.conn: ServerConnection | None = None
        self.connected = asyncio.Event()
        self.received: list[sp.SdkSignalFrame] = []
        self.offered_subprotocols: list[str] = []
        self.request_path: str | None = None
        self.bad_prefix_sent = False
        #: Toggles for individual tests.
        self.answer_join = True
        self.answer_subscribe = True
        self.answer_leave = True
        self.echo_data_messages = True
        self.subscribe_error: tuple[int, str] | None = None
        self.sdp_answer = ANSWER

    # -- lifecycle ------------------------------------------------------- #

    async def start(self) -> None:
        self.server = await serve(
            self._handler,
            "127.0.0.1",
            0,
            select_subprotocol=self._select_subprotocol,
            ping_interval=None,
        )
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    @property
    def signaling_url(self) -> str:
        # A query string is already present, to prove it is preserved.
        return f"ws://127.0.0.1:{self.port}/control/meeting-1?sessionId=abc"

    def _select_subprotocol(
        self, conn: ServerConnection, subprotocols: list[Subprotocol]
    ) -> Subprotocol | None:
        self.offered_subprotocols = list(subprotocols)
        return subprotocols[0] if subprotocols else None

    async def _handler(self, conn: ServerConnection) -> None:
        self.conn = conn
        self.request_path = conn.request.path if conn.request is not None else None
        self.connected.set()
        try:
            async for raw in conn:
                assert isinstance(raw, bytes)
                frame = bytes_to_frame(raw)
                self.received.append(frame)
                await self._on_frame(frame)
        except ConnectionClosed:
            pass

    # -- helpers --------------------------------------------------------- #

    async def send(self, frame: sp.SdkSignalFrame) -> None:
        assert self.conn is not None
        await self.conn.send(frame_to_bytes(frame))

    @staticmethod
    def _frame(frame_type: int) -> sp.SdkSignalFrame:
        frame = sp.SdkSignalFrame()
        frame.type = frame_type
        frame.timestamp_ms = 1
        return frame

    def frames_of(self, frame_type: int) -> list[sp.SdkSignalFrame]:
        return [f for f in self.received if f.type == frame_type]

    async def wait_for_frame(
        self, frame_type: int, timeout: float = 2.0
    ) -> sp.SdkSignalFrame:
        return (await self.wait_for_frames(frame_type, 1, timeout))[-1]

    async def wait_for_frames(
        self, frame_type: int, count: int, timeout: float = 2.0
    ) -> list[sp.SdkSignalFrame]:
        async with asyncio.timeout(timeout):
            while True:
                frames = self.frames_of(frame_type)
                if len(frames) >= count:
                    return frames
                await asyncio.sleep(0.01)

    async def _on_frame(self, frame: sp.SdkSignalFrame) -> None:
        if frame.type == sp.SdkSignalFrame.JOIN and self.answer_join:
            ack = self._frame(sp.SdkSignalFrame.JOIN_ACK)
            ack.joinack.video_subscription_limit = 25
            ack.joinack.turn_credentials.username = "turn-user"
            ack.joinack.turn_credentials.password = "turn-pass"
            ack.joinack.turn_credentials.ttl = 300
            ack.joinack.turn_credentials.uris.append("turn:127.0.0.1:3478")
            await self.send(ack)
        elif frame.type == sp.SdkSignalFrame.SUBSCRIBE and self.answer_subscribe:
            ack = self._frame(sp.SdkSignalFrame.SUBSCRIBE_ACK)
            if self.subscribe_error is not None:
                status, description = self.subscribe_error
                ack.error.status = status
                ack.error.description = description
            else:
                ack.suback.duplex = sp.SdkStreamServiceType.Value("RX")
                ack.suback.sdp_answer = self.sdp_answer
            await self.send(ack)
        elif frame.type == sp.SdkSignalFrame.LEAVE and self.answer_leave:
            await self.send(self._frame(sp.SdkSignalFrame.LEAVE_ACK))
        elif frame.type == sp.SdkSignalFrame.DATA_MESSAGE and self.echo_data_messages:
            echo = self._frame(sp.SdkSignalFrame.DATA_MESSAGE)
            for payload in frame.data_message.messages:
                out = echo.data_message.messages.add()
                out.topic = payload.topic
                out.data = payload.data
                out.lifetime_ms = payload.lifetime_ms
                out.sender_attendee_id = "attendee-other"
                out.sender_external_user_id = "E1-R1-U2"
                out.ingest_time_ns = 1234567890
            await self.send(echo)

    # -- things tests make the server do --------------------------------- #

    async def send_ping(self, ping_id: int) -> None:
        frame = self._frame(sp.SdkSignalFrame.PING_PONG)
        frame.ping_pong.type = sp.SdkPingPongType.Value("PING")
        frame.ping_pong.ping_id = ping_id
        await self.send(frame)

    async def send_index(self, sources: list[dict[str, Any]]) -> None:
        frame = self._frame(sp.SdkSignalFrame.INDEX)
        frame.index.num_participants = len(sources)
        for spec in sources:
            descriptor = frame.index.sources.add()
            for key, value in spec.items():
                setattr(descriptor, key, value)
        await self.send(frame)

    async def send_attendees(self, entries: list[dict[str, Any]]) -> None:
        frame = self._frame(sp.SdkSignalFrame.AUDIO_STREAM_ID_INFO)
        for spec in entries:
            info = frame.audio_stream_id_info.streams.add()
            for key, value in spec.items():
                setattr(info, key, value)
        await self.send(frame)

    async def send_raw(self, raw: bytes) -> None:
        assert self.conn is not None
        await self.conn.send(raw)

    async def close_connection(self, code: int, reason: str = "") -> None:
        assert self.conn is not None
        await self.conn.close(code=code, reason=reason)


def make_creds(signaling_url: str) -> ChimeCredentials:
    return ChimeCredentials(
        meeting_id="meeting-1",
        media_region="us-east-2",
        external_meeting_id="R0123456789",
        media_placement=MediaPlacement(
            audio_host_url="audio.example.invalid:3478",
            signaling_url=signaling_url,
        ),
        attendee_id=ATTENDEE,
        external_user_id="E1-R1-U1",
        join_token=JOIN_TOKEN,
    )


@pytest_asyncio.fixture
async def server():
    fake = FakeChimeServer()
    await fake.start()
    try:
        yield fake
    finally:
        await fake.stop()


@pytest_asyncio.fixture
async def client(server: FakeChimeServer):
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    try:
        yield sig
    finally:
        await sig.close()


# --------------------------------------------------------------------------- #
# frame codec / URL building
# --------------------------------------------------------------------------- #


def test_frame_round_trip() -> None:
    frame = sp.SdkSignalFrame()
    frame.type = sp.SdkSignalFrame.PING_PONG
    frame.timestamp_ms = 42
    frame.ping_pong.type = sp.SdkPingPongType.Value("PING")
    frame.ping_pong.ping_id = 7

    raw = frame_to_bytes(frame)
    assert raw[0] == 0x05
    back = bytes_to_frame(raw)
    assert back.type == sp.SdkSignalFrame.PING_PONG
    assert back.timestamp_ms == 42
    assert back.ping_pong.ping_id == 7


@pytest.mark.parametrize("raw", [b"", b"\x04\x08\x01", b"\x00"])
def test_bytes_to_frame_rejects_bad_prefix_in_strict_mode(raw: bytes) -> None:
    with pytest.raises(ValueError):
        bytes_to_frame(raw, strict=True)


def test_bytes_to_frame_rejects_empty_even_when_lenient() -> None:
    with pytest.raises(ValueError, match="empty"):
        bytes_to_frame(b"")


@pytest.mark.parametrize("prefix", sorted(RECEIVED_FRAME_PREFIXES))
def test_both_server_frame_prefixes_are_accepted(prefix: int) -> None:
    """Chime sends 0x02 as well as 0x05, and both carry an SdkSignalFrame.

    Regression test for a live failure: a real huddle sends a steady stream of
    0x02 frames, and rejecting them silently discarded INDEX and
    AUDIO_STREAM_ID_INFO -- so the attendee never became present and its
    drawings were never rendered. The real SDK's ``stripFrameTypeRTC()``
    accepts both, with the comment "TODO: change server frame type to send
    0x05".
    """
    frame = sp.SdkSignalFrame()
    frame.timestamp_ms = 1
    frame.type = sp.SdkSignalFrame.AUDIO_STREAM_ID_INFO
    info = frame.audio_stream_id_info.streams.add()
    info.audio_stream_id = 7
    info.attendee_id = "att-1"

    back = bytes_to_frame(bytes((prefix,)) + frame.SerializeToString())
    assert back.type == sp.SdkSignalFrame.AUDIO_STREAM_ID_INFO
    assert back.audio_stream_id_info.streams[0].attendee_id == "att-1"


def test_we_still_send_0x05() -> None:
    frame = sp.SdkSignalFrame()
    frame.timestamp_ms = 1
    frame.type = sp.SdkSignalFrame.JOIN
    assert frame_to_bytes(frame)[0] == 0x05


def test_unrecognised_prefix_is_decoded_anyway_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leniency is deliberate: the server has changed this byte once already."""
    frame = sp.SdkSignalFrame()
    frame.timestamp_ms = 1
    frame.type = sp.SdkSignalFrame.LEAVE_ACK
    with caplog.at_level("WARNING"):
        back = bytes_to_frame(b"\x09" + frame.SerializeToString())
    assert back.type == sp.SdkSignalFrame.LEAVE_ACK
    assert "unrecognised prefix" in caplog.text


def test_signaling_url_preserves_existing_query() -> None:
    url = signaling_url_for("wss://host.invalid/control/m1?sessionId=abc")
    assert url.startswith("wss://host.invalid/control/m1?sessionId=abc&")
    assert "X-Chime-Control-Protocol-Version=3" in url
    assert "X-Amzn-Chime-Send-Close-On-Error=1" in url

    plain = signaling_url_for("wss://host.invalid/control/m1")
    assert plain.endswith(
        "?X-Chime-Control-Protocol-Version=3&X-Amzn-Chime-Send-Close-On-Error=1"
    )


# --------------------------------------------------------------------------- #
# JOIN / JOIN_ACK
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_connect_handshake(server: FakeChimeServer) -> None:
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    try:
        # subprotocols: ['_aws_wt_session', <JoinToken>]
        assert server.offered_subprotocols == [SIGNALING_SUBPROTOCOL, JOIN_TOKEN]
        # the required query params, with the pre-existing one kept
        assert server.request_path is not None
        assert "sessionId=abc" in server.request_path
        assert "X-Chime-Control-Protocol-Version=3" in server.request_path
        assert "X-Amzn-Chime-Send-Close-On-Error=1" in server.request_path

        join = server.frames_of(sp.SdkSignalFrame.JOIN)[0]
        assert join.join.protocol_version == 2
        assert join.join.flags == sp.SdkJoinFlags.Value("HAS_STREAM_UPDATE") == 2
        assert join.join.client_details.app_name
        assert join.join.client_details.chime_sdk_version
        assert join.timestamp_ms > 0

        assert sig.turn_credentials is not None
        assert sig.turn_credentials.username == "turn-user"
        assert list(sig.turn_credentials.uris) == ["turn:127.0.0.1:3478"]
        assert sig.is_connected
        assert sig.attendee_id == ATTENDEE
    finally:
        await sig.close()


@pytest.mark.asyncio
async def test_connect_without_join_ack_raises_and_cleans_up(
    server: FakeChimeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.answer_join = False
    monkeypatch.setattr(signaling, "JOIN_ACK_TIMEOUT", 0.2)
    sig = ChimeSignaling(make_creds(server.signaling_url))
    with pytest.raises(SignalingError, match="JOIN_ACK"):
        await sig.connect()
    # connect() closed up after itself
    assert not sig.is_connected
    await sig.close()


# --------------------------------------------------------------------------- #
# PING / PONG
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_server_ping_is_answered_with_matching_pong(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await server.send_ping(4242)
    pong = await server.wait_for_frame(sp.SdkSignalFrame.PING_PONG)
    assert pong.ping_pong.type == sp.SdkPingPongType.Value("PONG")
    assert pong.ping_pong.ping_id == 4242
    assert pong.timestamp_ms > 0


@pytest.mark.asyncio
async def test_we_ping_the_server_periodically(
    server: FakeChimeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(signaling, "PING_INTERVAL", 0.05)
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    try:
        frames = await server.wait_for_frames(sp.SdkSignalFrame.PING_PONG, 2)
        assert all(
            f.ping_pong.type == sp.SdkPingPongType.Value("PING") for f in frames
        )
        # distinct, increasing ping ids
        ids = [f.ping_pong.ping_id for f in frames]
        assert ids == sorted(set(ids)) and ids[0] >= 1
    finally:
        await sig.close()


@pytest.mark.asyncio
async def test_unknown_prefix_byte_is_ignored(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await server.send_raw(b"\x01not-a-signal-frame")
    await server.send_raw(b"\x05\xff\xff\xff")  # right prefix, junk protobuf
    # the reader survives and still answers pings
    await server.send_ping(9)
    pong = await server.wait_for_frame(sp.SdkSignalFrame.PING_PONG)
    assert pong.ping_pong.ping_id == 9
    assert client.is_connected


# --------------------------------------------------------------------------- #
# INDEX
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_index_tracks_sources_and_content_share(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    seen: list[list[StreamSource]] = []
    unsubscribe = client.on_index(seen.append)

    await server.send_index(
        [
            {
                "stream_id": 2,
                "attendee_id": "attendee-other",
                "external_user_id": "E1-R1-U2",
                "media_type": sp.SdkStreamMediaType.Value("AUDIO"),
            },
            {
                "stream_id": 3,
                "attendee_id": "attendee-other#content",
                "external_user_id": "E1-R1-U2",
                "media_type": sp.SdkStreamMediaType.Value("VIDEO"),
                "width": 1920,
                "height": 1080,
                "group_id": 4,
                "max_bitrate_kbps": 1400,
            },
        ]
    )

    share = await client.wait_for_content_share(timeout=2.0)
    assert share.stream_id == 3
    assert share.is_content_share
    assert share.sharer_attendee_id == "attendee-other"
    assert share.media_type == "VIDEO"
    assert (share.width, share.height) == (1920, 1080)
    assert share.group_id == 4
    assert share.max_bitrate_kbps == 1400

    assert set(client.sources) == {2, 3}
    assert client.sources[2].media_type == "AUDIO"
    assert client.sources[2].external_user_id == "E1-R1-U2"
    assert not client.sources[2].is_content_share

    assert len(seen) == 1
    assert [s.stream_id for s in seen[0]] == [2, 3]

    # an already-known share returns immediately
    again = await client.wait_for_content_share(timeout=0.01)
    assert again.stream_id == 3

    unsubscribe()
    await server.send_index([])
    await asyncio.sleep(0.05)
    assert client.sources == {}
    assert len(seen) == 1  # unsubscribed


@pytest.mark.asyncio
async def test_index_is_a_full_replacement(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await server.send_index(
        [
            {
                "stream_id": 3,
                "attendee_id": "sharer-a#content",
                "media_type": sp.SdkStreamMediaType.Value("VIDEO"),
            }
        ]
    )
    first = await client.wait_for_content_share(timeout=2.0)
    assert first.sharer_attendee_id == "sharer-a"

    # the share restarted: new attendee id, new stream id, old entry gone
    await server.send_index(
        [
            {
                "stream_id": 9,
                "attendee_id": "sharer-b#content",
                "media_type": sp.SdkStreamMediaType.Value("VIDEO"),
            }
        ]
    )
    await asyncio.sleep(0.05)
    assert set(client.sources) == {9}
    assert client.sources[9].sharer_attendee_id == "sharer-b"


@pytest.mark.asyncio
async def test_wait_for_content_share_ignores_audio_only_sources(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await server.send_index(
        [
            {
                "stream_id": 1,
                "attendee_id": "attendee-other",
                "media_type": sp.SdkStreamMediaType.Value("AUDIO"),
            }
        ]
    )
    await asyncio.sleep(0.05)
    with pytest.raises(TimeoutError):
        await client.wait_for_content_share(timeout=0.1)


# --------------------------------------------------------------------------- #
# AUDIO_STREAM_ID_INFO
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_attendee_presence_updates_and_drops(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    batches: list[list[AttendeePresence]] = []
    client.on_attendees(batches.append)

    await server.send_attendees(
        [
            {
                "audio_stream_id": 11,
                "attendee_id": "attendee-other",
                "external_user_id": "E1-R1-U2",
                "muted": True,
            },
            {
                "audio_stream_id": 12,
                "attendee_id": "attendee-third",
                "external_user_id": "E1-R1-U3",
            },
        ]
    )

    presence = await client.wait_for_attendee("attendee-other", timeout=2.0)
    assert presence.audio_stream_id == 11
    assert presence.external_user_id == "E1-R1-U2"
    assert presence.muted is True
    assert set(client.attendees) == {"attendee-other", "attendee-third"}

    # an already-known attendee returns immediately
    assert (await client.wait_for_attendee("attendee-third", timeout=0.01)).muted is False

    # incremental: this frame mentions only one attendee, and drops it
    await server.send_attendees(
        [{"audio_stream_id": 12, "attendee_id": "attendee-third", "dropped": True}]
    )
    await asyncio.sleep(0.05)
    assert set(client.attendees) == {"attendee-other"}

    assert len(batches) == 2
    assert [e.attendee_id for e in batches[0]] == ["attendee-other", "attendee-third"]
    assert batches[1][0].dropped is True


@pytest.mark.asyncio
async def test_dropped_entry_without_attendee_id_still_removes(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await server.send_attendees(
        [{"audio_stream_id": 11, "attendee_id": "attendee-other"}]
    )
    await client.wait_for_attendee("attendee-other", timeout=2.0)
    await server.send_attendees([{"audio_stream_id": 11, "dropped": True}])
    await asyncio.sleep(0.05)
    assert client.attendees == {}


@pytest.mark.asyncio
async def test_wait_for_attendee_times_out(client: ChimeSignaling) -> None:
    with pytest.raises(TimeoutError):
        await client.wait_for_attendee("nobody", timeout=0.1)


# --------------------------------------------------------------------------- #
# SUBSCRIBE
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_subscribe_returns_answer_and_rewrites_origin_line(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    answer = await client.subscribe(
        sdp_offer=OFFER,
        audio_host="audio.example.invalid:3478",
        receive_stream_ids=(0, 3),
        audio_muted=True,
        send_audio_stream_id=1,
    )
    assert answer == ANSWER

    sub = server.frames_of(sp.SdkSignalFrame.SUBSCRIBE)[0].sub
    # the SDK rewrites o=- to o=mozilla-chrome
    assert "o=mozilla-chrome 4611731400430051336" in sub.sdp_offer
    assert "o=-" not in sub.sdp_offer
    assert sub.duplex == sp.SdkStreamServiceType.Value("RX")
    assert sub.audio_host == "audio.example.invalid:3478"
    assert sub.audio_muted is True
    assert list(sub.receive_stream_ids) == [0, 3]

    assert len(sub.send_streams) == 1
    stream = sub.send_streams[0]
    assert stream.media_type == sp.SdkStreamMediaType.Value("AUDIO")
    assert stream.attendee_id == ATTENDEE
    assert stream.stream_id == 1
    assert stream.track_label


@pytest.mark.asyncio
async def test_subscribe_leaves_a_mozilla_offer_alone(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    offer = OFFER.replace("o=-", "o=mozilla-firefox")
    await client.subscribe(sdp_offer=offer, audio_host="h")
    sub = server.frames_of(sp.SdkSignalFrame.SUBSCRIBE)[0].sub
    assert "o=mozilla-firefox" in sub.sdp_offer


@pytest.mark.asyncio
async def test_subscribe_error_mentions_the_video_section_hint(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    server.subscribe_error = (400, "failed to initialize video session")
    with pytest.raises(SignalingError) as excinfo:
        await client.subscribe(sdp_offer=OFFER, audio_host="h")
    message = str(excinfo.value)
    assert "failed to initialize video session" in message
    assert "recvonly video" in message


@pytest.mark.asyncio
async def test_subscribe_timeout_is_a_signaling_error(
    client: ChimeSignaling, server: FakeChimeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.answer_subscribe = False
    monkeypatch.setattr(signaling, "SUBSCRIBE_ACK_TIMEOUT", 0.2)
    with pytest.raises(SignalingError, match="SUBSCRIBE_ACK"):
        await client.subscribe(sdp_offer=OFFER, audio_host="h")


# --------------------------------------------------------------------------- #
# data messages
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_data_message_round_trip(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    received: list[DataMessage] = []

    async def on_message(message: DataMessage) -> None:  # async callback
        received.append(message)

    client.on_data_message(on_message)

    assert await client.send_data_message("data-channel", b"\x08\x01hello") is True

    sent = await server.wait_for_frame(sp.SdkSignalFrame.DATA_MESSAGE)
    assert len(sent.data_message.messages) == 1
    payload = sent.data_message.messages[0]
    assert payload.topic == "data-channel"
    assert payload.data == b"\x08\x01hello"
    assert payload.lifetime_ms == 0
    assert sent.timestamp_ms > 0

    async with asyncio.timeout(2.0):
        while not received:
            await asyncio.sleep(0.01)
    message = received[0]
    assert message.topic == "data-channel"
    assert message.data == b"\x08\x01hello"
    assert message.sender_attendee_id == "attendee-other"
    assert message.sender_external_user_id == "E1-R1-U2"
    assert message.ingest_time_ns == 1234567890


@pytest.mark.asyncio
async def test_data_message_lifetime_is_sent(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    await client.send_data_message("awareness-sync-channel", b"x", lifetime_ms=5000)
    sent = await server.wait_for_frame(sp.SdkSignalFrame.DATA_MESSAGE)
    assert sent.data_message.messages[0].lifetime_ms == 5000


@pytest.mark.asyncio
async def test_oversize_data_message_is_rejected(client: ChimeSignaling) -> None:
    with pytest.raises(ValueError, match="2048"):
        await client.send_data_message("data-channel", b"x" * 2049)
    # exactly at the limit is fine
    assert await client.send_data_message("data-channel", b"x" * 2048) is True


@pytest.mark.asyncio
async def test_rate_limited_send_with_wait_false_drops(
    server: FakeChimeServer,
) -> None:
    bucket = TokenBucket(rate=1.0, capacity=2.0)
    sig = ChimeSignaling(make_creds(server.signaling_url), rate_limiter=bucket)
    await sig.connect()
    try:
        assert await sig.send_data_message("data-channel", b"a", wait=False) is True
        assert await sig.send_data_message("data-channel", b"b", wait=False) is True
        # bucket is empty now; wait=False must return immediately, not block
        async with asyncio.timeout(0.5):
            assert await sig.send_data_message("data-channel", b"c", wait=False) is False
        sent = await server.wait_for_frames(sp.SdkSignalFrame.DATA_MESSAGE, 2)
        assert [m.data_message.messages[0].data for m in sent] == [b"a", b"b"]

        # wait=True blocks until the bucket refills, then sends
        async with asyncio.timeout(5.0):
            assert await sig.send_data_message("data-channel", b"d", wait=True) is True
        sent = await server.wait_for_frames(sp.SdkSignalFrame.DATA_MESSAGE, 3)
        assert [m.data_message.messages[0].data for m in sent] == [b"a", b"b", b"d"]
    finally:
        await sig.close()


# --------------------------------------------------------------------------- #
# callbacks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_raising_callback_does_not_kill_the_reader(
    client: ChimeSignaling, server: FakeChimeServer
) -> None:
    calls: list[str] = []

    def boom(_: list[AttendeePresence]) -> None:
        calls.append("boom")
        raise RuntimeError("callback blew up")

    async def also_boom(_: list[AttendeePresence]) -> None:
        calls.append("async-boom")
        raise RuntimeError("async callback blew up")

    def fine(_: list[AttendeePresence]) -> None:
        calls.append("fine")

    client.on_attendees(boom)
    client.on_attendees(also_boom)
    client.on_attendees(fine)

    await server.send_attendees(
        [{"audio_stream_id": 1, "attendee_id": "attendee-other"}]
    )
    await client.wait_for_attendee("attendee-other", timeout=2.0)
    assert calls == ["boom", "async-boom", "fine"]
    assert client.is_connected

    # and the session still works
    await server.send_ping(1)
    assert (await server.wait_for_frame(sp.SdkSignalFrame.PING_PONG)).ping_pong.ping_id == 1


@pytest.mark.asyncio
async def test_unsubscribe_is_idempotent(client: ChimeSignaling) -> None:
    unsubscribe = client.on_data_message(lambda _: None)
    unsubscribe()
    unsubscribe()  # must not raise


# --------------------------------------------------------------------------- #
# LEAVE / close / failure propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_leave_sends_leave_and_waits_for_ack(
    server: FakeChimeServer,
) -> None:
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    await sig.leave()
    assert server.frames_of(sp.SdkSignalFrame.LEAVE)
    await sig.close()
    assert not sig.is_connected


@pytest.mark.asyncio
async def test_leave_without_ack_does_not_hang(
    server: FakeChimeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.answer_leave = False
    monkeypatch.setattr(signaling, "LEAVE_ACK_TIMEOUT", 0.2)
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    async with asyncio.timeout(2.0):
        await sig.leave()  # logs, does not raise
    await sig.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_and_safe_after_an_error(
    server: FakeChimeServer,
) -> None:
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    await server.close_connection(4410, "meeting has ended")
    await asyncio.sleep(0.1)
    assert not sig.is_connected
    await sig.close()
    await sig.close()
    with pytest.raises(SignalingError):
        await sig.send_data_message("data-channel", b"x")


@pytest.mark.asyncio
async def test_close_before_connect() -> None:
    sig = ChimeSignaling(make_creds("ws://127.0.0.1:1/control/m"))
    await sig.close()


@pytest.mark.asyncio
async def test_connection_close_fails_pending_waiters_with_the_code(
    server: FakeChimeServer,
) -> None:
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    try:
        waiter = asyncio.ensure_future(sig.wait_for_attendee("nobody", timeout=10.0))
        await asyncio.sleep(0.05)
        await server.close_connection(4410, "meeting has ended")

        with pytest.raises(SignalingError) as excinfo:
            await waiter
        assert excinfo.value.code == 4410
        assert "the meeting has ended" in str(excinfo.value)
    finally:
        await sig.close()


@pytest.mark.asyncio
async def test_close_code_1006_is_explained(server: FakeChimeServer) -> None:
    sig = ChimeSignaling(make_creds(server.signaling_url))
    await sig.connect()
    try:
        # abort without a close handshake -> the client sees 1006
        assert server.conn is not None
        server.conn.transport.abort()
        async with asyncio.timeout(2.0):
            while sig.is_connected:
                await asyncio.sleep(0.01)
        with pytest.raises(SignalingError) as excinfo:
            await sig.send_data_message("data-channel", b"x")
        assert excinfo.value.code == 1006
        assert "rate limit" in str(excinfo.value)
    finally:
        await sig.close()


@pytest.mark.asyncio
async def test_connect_refused_raises_signaling_error() -> None:
    sig = ChimeSignaling(make_creds("ws://127.0.0.1:1/control/m"))
    with pytest.raises(SignalingError, match="could not open"):
        await sig.connect()


@pytest.mark.asyncio
async def test_double_connect_is_rejected(client: ChimeSignaling) -> None:
    with pytest.raises(SignalingError, match="already connected"):
        await client.connect()
