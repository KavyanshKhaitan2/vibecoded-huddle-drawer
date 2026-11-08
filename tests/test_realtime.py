"""Tests for :mod:`huddle_drawings.realtime`.

No network: every test that needs a gateway runs a local ``websockets`` server
(:class:`FakeGateway`) on ``127.0.0.1`` and points the client at it by
monkeypatching ``realtime.GATEWAY_BASE_URL``. No real Slack identity is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from huddle_drawings import realtime
from huddle_drawings.config import Config
from huddle_drawings.realtime import (
    START_ARGS,
    HuddleInvite,
    RealtimeClient,
    RealtimeError,
    RoomEvent,
    build_gateway_url,
    extract_room_id,
    gateway_server_for,
    reconnect_delay,
)

XOXC = "xoxc-test-0000000000-deadbeefcafe"
XOXD = "xoxd-test-1111111111-feedfacebead"
TEAM_ID = "E0123ABCD"
JOIN_TOKEN = "join-token-secret-abcdef"

FREE_WILLY: dict[str, Any] = {
    "meeting": {
        "MeetingId": "11111111-2222-3333-4444-555555555555",
        "MediaRegion": "us-east-2",
        "ExternalMeetingId": "R0123456789",
        "MeetingFeatures": None,
        "MediaPlacement": {
            "AudioHostUrl": "audio.m3.ue2.app.chime.aws:3478",
            "SignalingUrl": "wss://signal.m3.ue2.app.chime.aws/control/meeting",
            "TurnControlUrl": "https://ccp.cp.ue2.app.chime.aws/v2/turn_sessions",
        },
    },
    "attendee": {
        "AttendeeId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "ExternalUserId": "E0123ABCD-R0123456789-U0999",
        "JoinToken": JOIN_TOKEN,
    },
}


def make_config(**overrides: Any) -> Config:
    kwargs: dict[str, Any] = {
        "xoxc_token": XOXC,
        "xoxd_token": XOXD,
        "slack_host": "hackclub.slack.com",
    }
    kwargs.update(overrides)
    return Config(**kwargs)


# --------------------------------------------------------------------------- #
# a local stand-in for wss-primary.slack.com
# --------------------------------------------------------------------------- #


class FakeGateway:
    """A minimal local websocket server that behaves like the flannel gateway."""

    def __init__(self, *, auto_hello: bool = True, close_on_connect: bool = False) -> None:
        self.auto_hello = auto_hello
        self.close_on_connect = close_on_connect
        #: ``(path, headers)`` per accepted connection, in order. ``headers``
        #: is websockets' case-insensitive ``Headers`` mapping.
        self.requests: list[tuple[str, Any]] = []
        #: Text frames the client sent us.
        self.received: asyncio.Queue[str] = asyncio.Queue()
        #: Each accepted connection, in order.
        self.connections: asyncio.Queue[ServerConnection] = asyncio.Queue()
        self._server: Server | None = None
        self.port = 0

    async def start(self) -> None:
        self._server = await serve(self._handler, "127.0.0.1", 0)
        self.port = next(iter(self._server.sockets)).getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def base_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/"

    async def _handler(self, ws: ServerConnection) -> None:
        self.requests.append((ws.request.path, ws.request.headers))
        await self.connections.put(ws)
        if self.close_on_connect:
            await ws.close()
            return
        if self.auto_hello:
            await ws.send(json.dumps({"type": "hello"}))
        try:
            async for message in ws:
                await self.received.put(
                    message if isinstance(message, str) else message.decode()
                )
        except ConnectionClosed:
            pass

    async def next_connection(self, timeout: float = 5.0) -> ServerConnection:
        return await asyncio.wait_for(self.connections.get(), timeout)

    async def next_message(self, timeout: float = 5.0) -> dict[str, Any]:
        raw = await asyncio.wait_for(self.received.get(), timeout)
        return json.loads(raw)


@pytest.fixture
async def gateway():
    server = FakeGateway()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
def local_gateway(gateway, monkeypatch):
    """``gateway``, with the module's base URL pointed at it."""
    monkeypatch.setattr(realtime, "GATEWAY_BASE_URL", gateway.base_url)
    return gateway


@pytest.fixture
async def make_client(local_gateway):
    """Factory for clients wired to the local gateway; closed on teardown."""
    clients: list[RealtimeClient] = []

    def make(**kwargs: Any) -> RealtimeClient:
        kwargs.setdefault("team_id", TEAM_ID)
        client = RealtimeClient(make_config(), **kwargs)
        clients.append(client)
        return client

    try:
        yield make
    finally:
        for client in clients:
            await client.close()


@pytest.fixture
async def client(make_client) -> RealtimeClient:
    return make_client()


async def arm(
    client: RealtimeClient, type_: str, **kwargs: Any
) -> asyncio.Task[dict[str, Any]]:
    """Start waiting for an event *before* the server sends it.

    Sending first and waiting afterwards is racy: the reader can dispatch the
    frame before the waiting coroutine has had a turn to subscribe.
    """
    kwargs.setdefault("timeout", 5)
    task = asyncio.create_task(client.wait_for_event(type_, **kwargs))
    await asyncio.sleep(0)
    return task


async def wait_until(predicate, timeout: float = 5.0) -> None:
    """Poll ``predicate`` until it is true (for state the client sets itself)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #


def test_desktop_url_has_every_documented_parameter() -> None:
    url = build_gateway_url(XOXC, team_id=TEAM_ID, enterprise_id=TEAM_ID)
    parts = urlsplit(url)
    assert parts.scheme == "wss"
    assert parts.netloc == "wss-primary.slack.com"

    query = parse_qs(parts.query, keep_blank_values=True)
    assert query["token"] == [XOXC]
    assert query["sync_desync"] == ["1"]
    assert query["slack_client"] == ["desktop"]
    assert query["no_query_on_subscribe"] == ["1"]
    assert query["flannel"] == ["3"]
    assert query["lazy_channels"] == ["1"]
    assert query["enterprise_id"] == [TEAM_ID]
    assert query["batch_presence_aware"] == ["1"]
    # Order matters only cosmetically, but keep the doc's order.
    assert [key for key, _ in [p.split("=", 1) for p in parts.query.split("&")]] == [
        "token",
        "sync_desync",
        "slack_client",
        "start_args",
        "no_query_on_subscribe",
        "flannel",
        "lazy_channels",
        "gateway_server",
        "enterprise_id",
        "batch_presence_aware",
    ]


def test_start_args_is_percent_encoded_as_a_single_value() -> None:
    url = build_gateway_url(XOXC, team_id=TEAM_ID)
    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    # One parameter, whose value is the whole nested query string verbatim.
    assert query["start_args"] == [START_ARGS]
    assert START_ARGS.startswith("?")
    # Percent-encoded, so the nested '?', '&' and '=' cannot leak into the
    # outer query string.
    assert "start_args=%3Fagent%3Dclient%26org_wide_aware%3Dtrue" in url
    assert "agent=client" not in url
    assert "agent" not in query
    assert "org_wide_aware" not in query


def test_gateway_server_derivations() -> None:
    # huddlefm: first character of the team id replaced by 'T', suffix '-1'.
    assert gateway_server_for("E0123ABCD") == "T0123ABCD-1"
    assert gateway_server_for("T0123ABCD") == "T0123ABCD-1"
    # super-platinum: the plain team id, suffix '-1'.
    assert gateway_server_for("E0123ABCD", simple=True) == "E0123ABCD-1"
    with pytest.raises(ValueError):
        gateway_server_for("")
    with pytest.raises(ValueError):
        gateway_server_for("E")


def test_simple_url_is_the_super_platinum_form() -> None:
    url = build_gateway_url(XOXC, team_id=TEAM_ID, enterprise_id=TEAM_ID, simple=True)
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    assert set(query) == {"token", "flannel", "gateway_server"}
    assert query["token"] == [XOXC]
    assert query["flannel"] == ["3"]
    assert query["gateway_server"] == [f"{TEAM_ID}-1"]
    # The simple form carries neither start_args nor enterprise_id.
    assert "start_args" not in url
    assert "enterprise_id" not in url


def test_client_url_properties_follow_simple_url_flag() -> None:
    config = make_config()
    desktop = RealtimeClient(config, team_id=TEAM_ID, enterprise_id=TEAM_ID)
    simple = RealtimeClient(config, team_id=TEAM_ID, simple_url=True)
    assert desktop.gateway_server == "T0123ABCD-1"
    assert simple.gateway_server == "E0123ABCD-1"
    assert "sync_desync=1" in desktop.url
    assert "sync_desync" not in simple.url

    # The token is in the URL (the gateway needs it there) but never in the
    # loggable form, and the d cookie never rides in the query string.
    assert XOXC in desktop.url
    assert XOXD not in desktop.url
    assert XOXC not in desktop.safe_url
    assert "token=<redacted>" in desktop.safe_url
    assert "gateway_server=T0123ABCD-1" in desktop.safe_url


def test_empty_team_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        RealtimeClient(make_config(), team_id="")


def test_reconnect_delay_is_exponential_and_capped_at_30s() -> None:
    assert all(0.0 < reconnect_delay(i) <= 30.0 for i in range(20))
    # Monotone ceilings: the 0th attempt can never be as slow as a late one.
    assert reconnect_delay(0) <= realtime.RECONNECT_BACKOFF_BASE
    assert reconnect_delay(12) > 10.0


# --------------------------------------------------------------------------- #
# event models
# --------------------------------------------------------------------------- #


def test_extract_room_id_handles_all_four_payload_shapes() -> None:
    assert extract_room_id({"type": "sh_room_join", "call_id": "R1"}) == "R1"
    assert extract_room_id({"type": "sh_room_join", "room": {"call_id": "R2"}}) == "R2"
    # super-platinum's fixture shape.
    assert (
        extract_room_id(
            {
                "type": "sh_room_join",
                "user": "U1",
                "huddle": {"channel_id": "C1"},
                "room": {"id": "R3", "call_family": "huddle", "channels": ["C1"]},
            }
        )
        == "R3"
    )
    assert extract_room_id({"type": "sh_room_update", "huddle": {"id": "R4"}}) == "R4"
    # Precedence: top-level call_id wins, then room.call_id, then room.id.
    assert (
        extract_room_id({"call_id": "R1", "room": {"call_id": "R2", "id": "R3"}}) == "R1"
    )
    assert extract_room_id({"room": {"call_id": "R2", "id": "R3"}}) == "R2"
    # Nothing usable, and junk, both give None instead of raising.
    assert extract_room_id({"type": "sh_room_join"}) is None
    assert extract_room_id({"room": "not-a-dict", "huddle": None}) is None
    assert extract_room_id({"call_id": ""}) is None


def test_room_event_from_event_super_platinum_fixture() -> None:
    event = {
        "type": "sh_room_join",
        "user": "U0999",
        "huddle": {"channel_id": "C0123"},
        "room": {
            "id": "R0123456789",
            "call_family": "huddle",
            "channels": ["C0123"],
            "created_by": "U0999",
            "has_ended": False,
            "participants": ["U0999", {"user_id": "U0888"}],
            "media_backend_type": "free_willy",
        },
    }
    room = RoomEvent.from_event(event)
    assert room.type == "sh_room_join"
    assert room.call_id == "R0123456789"
    assert room.user == "U0999"
    assert room.channel_id == "C0123"
    assert room.participants == ("U0999", "U0888")
    assert room.ended is False
    assert room.raw is event


def test_room_event_ends_on_either_marker() -> None:
    assert RoomEvent.from_event(
        {"type": "sh_room_update", "huddle": {"id": "R1", "has_ended": True}}
    ).ended
    assert RoomEvent.from_event(
        {"type": "sh_room_update", "room": {"id": "R1"}, "huddle": {"date_end": 1764000000}}
    ).ended
    live = RoomEvent.from_event(
        {"type": "sh_room_update", "room": {"id": "R1", "has_ended": False, "date_end": 0}}
    )
    assert live.ended is False
    assert live.date_end is None


def test_huddle_thread_message_carries_a_room() -> None:
    event = {
        "type": "message",
        "subtype": realtime.MESSAGE_SUBTYPE_HUDDLE_THREAD,
        "channel": "C0123",
        "room": {"id": "R0123", "participants": ["U1"], "has_ended": False},
    }
    room = RoomEvent.from_event(event)
    assert room.call_id == "R0123"
    assert room.channel_id == "C0123"
    assert room.participants == ("U1",)


def test_huddle_invite_keeps_the_free_willy_blob() -> None:
    event = {
        "type": "huddle_invite",
        "channel_id": "C0123",
        "call_id": "R0123456789",
        "sender_user_id": "U0888",
        "free_willy": FREE_WILLY,
    }
    invite = HuddleInvite.from_event(event)
    assert invite.channel_id == "C0123"
    assert invite.call_id == "R0123456789"
    assert invite.sender_user_id == "U0888"
    # Intact and unmodified -- this is what makes joining possible with no
    # second API call.
    assert invite.free_willy == FREE_WILLY
    assert invite.has_chime_credentials
    assert invite.raw is event

    creds = invite.chime_credentials()
    assert creds.meeting_id == FREE_WILLY["meeting"]["MeetingId"]
    assert creds.attendee_id == FREE_WILLY["attendee"]["AttendeeId"]
    assert creds.join_token == JOIN_TOKEN
    assert creds.media_placement.signaling_url.startswith("wss://")
    # A null MeetingFeatures is stripped, as the Chime SDK requires.
    assert "MeetingFeatures" not in creds.raw["meeting"]
    assert FREE_WILLY["meeting"]["MeetingFeatures"] is None  # input untouched
    # Secrets stay out of reprs.
    assert JOIN_TOKEN not in repr(creds)
    assert JOIN_TOKEN not in repr(invite)


def test_huddle_invite_without_a_blob_is_explicit() -> None:
    invite = HuddleInvite.from_event({"type": "huddle_invite", "channel_id": "C1"})
    assert invite.free_willy is None
    assert not invite.has_chime_credentials
    with pytest.raises(RealtimeError):
        invite.chime_credentials()

    broken = HuddleInvite.from_event(
        {"type": "huddle_invite", "free_willy": {"meeting": {}, "attendee": {}}}
    )
    assert broken.has_chime_credentials is False
    with pytest.raises(RealtimeError):
        broken.chime_credentials()


# --------------------------------------------------------------------------- #
# handshake, pings, reconnect
# --------------------------------------------------------------------------- #


async def test_connect_waits_for_hello_and_sends_the_cookie(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    assert not client.is_connected
    await client.connect()
    assert client.is_connected

    path, headers = local_gateway.requests[0]
    assert f"token={XOXC}" in path
    assert "gateway_server=T0123ABCD-1" in path
    # The d cookie goes in a raw Cookie header, never in the query string.
    assert headers["Cookie"] == f"d={XOXD}"
    assert XOXD not in path

    await client.close()
    assert not client.is_connected


async def test_connecting_twice_is_an_error(client: RealtimeClient) -> None:
    await client.connect()
    with pytest.raises(RealtimeError):
        await client.connect()


async def test_no_hello_means_no_connection(
    make_client, local_gateway: FakeGateway, monkeypatch
) -> None:
    monkeypatch.setattr(realtime, "HELLO_TIMEOUT", 0.2)
    local_gateway.auto_hello = False
    client = make_client()
    with pytest.raises(RealtimeError, match="no hello"):
        await client.connect()
    assert not client.is_connected
    # The failed attempt left nothing behind, so a retry is possible.
    local_gateway.auto_hello = True
    await client.connect()
    assert client.is_connected


async def test_close_during_the_handshake_is_reported(
    make_client, local_gateway: FakeGateway
) -> None:
    local_gateway.auto_hello = False
    local_gateway.close_on_connect = True
    client = make_client()
    with pytest.raises(RealtimeError, match="hello handshake"):
        await client.connect()
    assert not client.is_connected


async def test_the_url_token_is_scrubbed_from_logs(
    client: RealtimeClient, local_gateway: FakeGateway, caplog
) -> None:
    caplog.set_level(logging.DEBUG, logger="huddle_drawings.realtime")
    await client.connect()
    reconnect_url = f"{local_gateway.base_url}?token=xoxc-another-secret&flannel=3"
    connection = await local_gateway.next_connection()

    waiter = await arm(client, "reconnect_url")
    await connection.send(json.dumps({"type": "reconnect_url", "url": reconnect_url}))
    await waiter

    waiter = await arm(client, "huddle_invite")
    await connection.send(
        json.dumps(
            {
                "type": "huddle_invite",
                "channel_id": "C1",
                "call_id": "R1",
                "sender_user_id": "U1",
                "free_willy": FREE_WILLY,
            }
        )
    )
    await waiter
    await client.close()

    assert "huddle_invite" in caplog.text  # we really did log about these
    assert "<redacted>" in caplog.text
    assert XOXC not in caplog.text
    assert XOXD not in caplog.text
    assert "xoxc-another-secret" not in caplog.text
    assert JOIN_TOKEN not in caplog.text


async def test_pings_are_sent_every_interval(
    make_client, local_gateway: FakeGateway, monkeypatch
) -> None:
    monkeypatch.setattr(realtime, "PING_INTERVAL", 0.05)
    client = make_client()
    await client.connect()

    first = await local_gateway.next_message()
    second = await local_gateway.next_message()
    assert first["type"] == "ping"
    assert second["type"] == "ping"
    # ``{"type":"ping","id":<ms>}``: milliseconds, and always increasing.
    assert isinstance(first["id"], int)
    assert first["id"] > 1_700_000_000_000
    assert second["id"] > first["id"]


async def test_reconnect_url_is_stored_and_used_for_the_next_reconnect(
    make_client, local_gateway: FakeGateway, monkeypatch
) -> None:
    monkeypatch.setattr(realtime, "RECONNECT_BACKOFF_BASE", 0.01)
    client = make_client()
    await client.connect()
    first = await local_gateway.next_connection()

    reconnect_url = f"{local_gateway.base_url}?reconnect=1&token={XOXC}"
    waiter = await arm(client, "reconnect_url")
    await first.send(json.dumps({"type": "reconnect_url", "url": reconnect_url}))
    await waiter
    assert client.reconnect_url == reconnect_url

    runner = asyncio.create_task(client.run_forever())
    try:
        await first.close()
        second = await local_gateway.next_connection()
        await wait_until(lambda: client.is_connected)
        assert "reconnect=1" in local_gateway.requests[1][0]
        # Consumed: a stale reconnect_url must not be retried forever.
        assert client.reconnect_url is None
        assert second.request is not None
    finally:
        # close() is what stops run_forever; it must not reconnect again.
        await client.close()
        await asyncio.wait_for(runner, 5)


async def test_run_forever_reconnects_with_the_plain_url(
    make_client, local_gateway: FakeGateway, monkeypatch
) -> None:
    monkeypatch.setattr(realtime, "RECONNECT_BACKOFF_BASE", 0.01)
    client = make_client()
    await client.connect()
    first = await local_gateway.next_connection()

    runner = asyncio.create_task(client.run_forever())
    try:
        await first.close()
        await local_gateway.next_connection()
        await wait_until(lambda: client.is_connected)
        assert len(local_gateway.requests) == 2
        assert "sync_desync=1" in local_gateway.requests[1][0]
    finally:
        # close() is what stops run_forever; it must not reconnect again.
        await client.close()
        await asyncio.wait_for(runner, 5)
    # close() stops the loop rather than reconnecting again.
    assert not client.is_connected


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


async def test_huddle_invite_is_dispatched_with_the_blob_intact(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    invites: list[HuddleInvite] = []
    client.on_huddle_invite(invites.append)
    seen: list[dict[str, Any]] = []
    client.on_event("*", seen.append)

    await client.connect()
    connection = await local_gateway.next_connection()
    event = {
        "type": "huddle_invite",
        "channel_id": "C0123",
        "call_id": "R0123456789",
        "sender_user_id": "U0888",
        "free_willy": FREE_WILLY,
    }
    waiter = await arm(client, "huddle_invite")
    await connection.send(json.dumps(event))
    received = await waiter

    assert received["free_willy"] == FREE_WILLY
    assert len(invites) == 1
    invite = invites[0]
    assert invite.call_id == "R0123456789"
    assert invite.chime_credentials().join_token == JOIN_TOKEN
    # The wildcard subscription saw the hello too.
    assert [e.get("type") for e in seen] == ["hello", "huddle_invite"]


async def test_unknown_and_malformed_frames_do_not_kill_the_reader(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    await client.connect()
    connection = await local_gateway.next_connection()

    for frame in (
        "this is not json",
        "[1, 2, 3]",
        '"a bare string"',
        "{}",  # no "type"
        json.dumps({"type": "something_we_have_never_seen", "x": 1}),
        json.dumps({"type": "sh_room_join", "room": "not-a-dict"}),
        b"\x80\x81\x82",  # undecodable binary
    ):
        await connection.send(frame)

    # The junk above includes an sh_room_join with an unusable payload, so
    # match on the id rather than on the type alone.
    waiter = await arm(
        client, "sh_room_join", predicate=lambda e: extract_room_id(e) == "R9"
    )
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R9"}}))
    event = await waiter
    assert RoomEvent.from_event(event).call_id == "R9"
    assert client.is_connected


async def test_a_raising_callback_does_not_kill_the_reader(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    calls: list[str] = []

    def boom(event: dict[str, Any]) -> None:
        calls.append("sync")
        raise RuntimeError("callback blew up")

    async def boom_async(event: dict[str, Any]) -> None:
        calls.append("async")
        raise RuntimeError("async callback blew up")

    def good(event: dict[str, Any]) -> None:
        calls.append("good")

    client.on_event("sh_room_join", boom)
    client.on_event(["sh_room_join", "sh_room_leave"], boom_async)
    client.on_event("sh_room_join", good)

    await client.connect()
    connection = await local_gateway.next_connection()
    waiter = await arm(client, "sh_room_join")
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R1"}}))
    await waiter
    assert calls == ["sync", "async", "good"]

    # Still alive for the next event.
    waiter = await arm(client, "sh_room_join")
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R2"}}))
    event = await waiter
    assert extract_room_id(event) == "R2"
    assert calls.count("good") == 2


async def test_on_event_accepts_one_type_several_or_a_wildcard(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    one: list[str] = []
    several: list[str] = []
    everything: list[str] = []
    unsubscribe = client.on_event("sh_room_join", lambda e: one.append(e["type"]))
    client.on_event(
        ("sh_room_join", "sh_room_leave"), lambda e: several.append(e["type"])
    )
    client.on_event("*", lambda e: everything.append(e.get("type", "")))

    await client.connect()
    connection = await local_gateway.next_connection()
    waiter = await arm(client, "sh_room_join")
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R1"}}))
    await waiter

    unsubscribe()
    unsubscribe()  # idempotent
    waiter = await arm(client, "sh_room_leave")
    await connection.send(json.dumps({"type": "sh_room_leave", "room": {"id": "R1"}}))
    await waiter

    assert one == ["sh_room_join"]
    assert several == ["sh_room_join", "sh_room_leave"]
    assert everything == ["hello", "sh_room_join", "sh_room_leave"]

    with pytest.raises(ValueError):
        client.on_event([], lambda e: None)


async def test_wait_for_event_with_a_predicate(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    await client.connect()
    connection = await local_gateway.next_connection()

    waiter = asyncio.create_task(
        client.wait_for_event(
            "sh_room_join",
            timeout=5,
            predicate=lambda e: e.get("user") == "U_WANTED",
        )
    )
    await asyncio.sleep(0)
    await connection.send(
        json.dumps({"type": "sh_room_join", "user": "U_OTHER", "room": {"id": "R1"}})
    )
    await connection.send(
        json.dumps({"type": "sh_room_join", "user": "U_WANTED", "room": {"id": "R2"}})
    )
    event = await waiter
    assert event["user"] == "U_WANTED"
    assert extract_room_id(event) == "R2"


async def test_wait_for_event_times_out(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    await client.connect()
    with pytest.raises(TimeoutError):
        await client.wait_for_event("huddle_invite", timeout=0.1)
    assert client.is_connected


async def test_wait_for_event_propagates_a_broken_predicate(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    await client.connect()
    connection = await local_gateway.next_connection()

    def predicate(event: dict[str, Any]) -> bool:
        raise RuntimeError("bad predicate")

    waiter = asyncio.create_task(
        client.wait_for_event("sh_room_join", timeout=5, predicate=predicate)
    )
    await asyncio.sleep(0)
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R1"}}))
    with pytest.raises(RuntimeError, match="bad predicate"):
        await waiter
    assert client.is_connected


# --------------------------------------------------------------------------- #
# close
# --------------------------------------------------------------------------- #


async def test_close_is_idempotent(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    await client.connect()
    waiter = asyncio.create_task(client.wait_for_event("huddle_invite", timeout=5))
    await asyncio.sleep(0)

    await client.close()
    await client.close()
    await client.close()

    assert client.is_closed
    assert not client.is_connected
    with pytest.raises(RealtimeError):
        await waiter
    with pytest.raises(RealtimeError):
        await client.wait_for_event("hello", timeout=0.1)
    with pytest.raises(RealtimeError):
        await client.connect()


async def test_close_without_connecting_is_fine(client: RealtimeClient) -> None:
    await client.close()
    await client.close()
    assert client.is_closed


async def test_close_from_inside_a_callback(
    client: RealtimeClient, local_gateway: FakeGateway
) -> None:
    closed = asyncio.Event()

    async def on_room(event: dict[str, Any]) -> None:
        await client.close()
        closed.set()

    client.on_event("sh_room_join", on_room)
    await client.connect()
    connection = await local_gateway.next_connection()
    await connection.send(json.dumps({"type": "sh_room_join", "room": {"id": "R1"}}))
    await asyncio.wait_for(closed.wait(), 5)
    assert client.is_closed


async def test_context_manager_connects_and_closes(
    make_client, local_gateway: FakeGateway
) -> None:
    client = make_client()
    async with client as entered:
        assert entered is client
        assert client.is_connected
    assert client.is_closed


async def test_close_cuts_a_reconnect_backoff_short(
    make_client, local_gateway: FakeGateway, monkeypatch
) -> None:
    # A 30s first backoff: close() must not wait for it.
    monkeypatch.setattr(realtime, "RECONNECT_BACKOFF_BASE", 30.0)
    client = make_client()
    await client.connect()
    first = await local_gateway.next_connection()
    await local_gateway.stop()  # nothing to reconnect to

    runner = asyncio.create_task(client.run_forever())
    await first.close()
    await wait_until(lambda: not client.is_connected)
    await client.close()
    await asyncio.wait_for(runner, 5)


async def test_run_forever_returns_after_close(
    make_client, local_gateway: FakeGateway
) -> None:
    client = make_client()
    await client.connect()
    runner = asyncio.create_task(client.run_forever())
    await asyncio.sleep(0.05)
    await client.close()
    await asyncio.wait_for(runner, 5)
    assert len(local_gateway.requests) == 1


# --------------------------------------------------------------------------- #
# from_slack
# --------------------------------------------------------------------------- #


class FakeSlack:
    """Just enough of ``SlackClient`` for ``from_slack``."""

    def __init__(self, auth: dict[str, Any]) -> None:
        self._auth = auth
        self.calls = 0

    async def auth_test(self) -> dict[str, Any]:
        self.calls += 1
        return self._auth


async def test_from_slack_derives_ids_from_auth_test() -> None:
    slack = FakeSlack({"ok": True, "team_id": TEAM_ID, "user_id": "U0999"})
    client = await RealtimeClient.from_slack(make_config(), slack)  # type: ignore[arg-type]
    assert slack.calls == 1
    assert client.team_id == TEAM_ID
    # An 'E…' team id *is* the enterprise id (see the from_slack docstring).
    assert client.enterprise_id == TEAM_ID
    assert client.gateway_server == "T0123ABCD-1"


async def test_from_slack_prefers_an_explicit_enterprise_id() -> None:
    slack = FakeSlack(
        {"ok": True, "team_id": "T0123ABCD", "enterprise_id": "E7777777"}
    )
    client = await RealtimeClient.from_slack(make_config(), slack)  # type: ignore[arg-type]
    assert client.enterprise_id == "E7777777"
    assert "enterprise_id=E7777777" in client.url

    nested = FakeSlack(
        {"ok": True, "team_id": "T0123ABCD", "enterprise": {"id": "E8888888"}}
    )
    client = await RealtimeClient.from_slack(make_config(), nested)  # type: ignore[arg-type]
    assert client.enterprise_id == "E8888888"


async def test_from_slack_passes_kwargs_through() -> None:
    slack = FakeSlack({"ok": True, "team_id": TEAM_ID})
    client = await RealtimeClient.from_slack(
        make_config(), slack, simple_url=True, enterprise_id=None  # type: ignore[arg-type]
    )
    assert client.gateway_server == f"{TEAM_ID}-1"
    assert "enterprise_id" not in client.url


async def test_from_slack_without_a_team_id_raises() -> None:
    slack = FakeSlack({"ok": True, "user_id": "U1"})
    with pytest.raises(RealtimeError, match="team_id"):
        await RealtimeClient.from_slack(make_config(), slack)  # type: ignore[arg-type]
