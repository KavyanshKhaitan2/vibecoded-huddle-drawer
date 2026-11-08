"""Tests for :mod:`huddle_drawings.slack`.

Everything runs against ``httpx.MockTransport`` -- no network, and no real
Slack identity is ever used.
"""

from __future__ import annotations

import asyncio
import copy
from email import message_from_bytes
from typing import Any

import httpx
import pytest

from huddle_drawings.config import Config
from huddle_drawings.slack import (
    ACCEPT_INVITE_VALUES,
    INVITE_METHOD_CANDIDATES,
    SlackClient,
    normalise_participants,
    parse_room_info,
    parse_rooms_join,
    strip_null_meeting_features,
)
from huddle_drawings.types import HuddleError, SlackApiError

XOXC = "xoxc-test-0000000000-deadbeefcafe"
XOXD = "xoxd-test-1111111111-feedfacebead"
JOIN_TOKEN = "join-token-secret-abcdef"


def make_config(**overrides: Any) -> Config:
    kwargs: dict[str, Any] = {
        "xoxc_token": XOXC,
        "xoxd_token": XOXD,
        "slack_host": "hackclub.slack.com",
        "media_region": "us-east-2",
        "user_agent": "insomnia/11.2.0",
    }
    kwargs.update(overrides)
    return Config(**kwargs)


def parse_multipart(request: httpx.Request) -> dict[str, str]:
    """Decode a multipart/form-data body into ``{field: value}``."""
    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n"
    raw += request.content
    message = message_from_bytes(raw)
    assert message.is_multipart(), "request body was not multipart"
    fields: dict[str, str] = {}
    for part in message.get_payload():
        name = part.get_param("name", header="content-disposition")
        assert isinstance(name, str)
        fields[name] = part.get_payload(decode=True).decode()
    return fields


def client_for(
    handler: Any, config: Config | None = None
) -> tuple[SlackClient, Config]:
    cfg = config or make_config()
    transport = httpx.MockTransport(handler)
    return SlackClient(cfg, client=httpx.AsyncClient(transport=transport)), cfg


def json_handler(payload: dict[str, Any], *, record: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        return httpx.Response(200, json=payload)

    return handler


# -- fixtures ---------------------------------------------------------------


def rooms_join_payload(*, participants: Any = None, meeting_features: Any = None) -> dict[str, Any]:
    """A realistic ``rooms.join`` response, shaped as the doc records it."""
    return {
        "ok": True,
        "call": {
            "call_id": "R09ABCDEF12",
            "survey_percent": 0,
            "free_willy": {
                "meeting": {
                    "MeetingId": "3f9c1e8a-2b7d-4c5e-9a10-7b6c5d4e3f21",
                    "MediaRegion": "us-east-2",
                    "ExternalMeetingId": "E012ABCDEF-R09ABCDEF12",
                    "MeetingFeatures": meeting_features,
                    "MediaPlacement": {
                        "AudioHostUrl": "aud.m3.ue2.app.chime.aws:3478",
                        "AudioFallbackUrl": "wss://haxrp.m3.ue2.app.chime.aws:443/calls/x",
                        "SignalingUrl": "wss://signal.m3.ue2.app.chime.aws/control/x",
                        "TurnControlUrl": "https://ccp.cp.ue2.app.chime.aws/v2/turn_sessions",
                        "ScreenDataUrl": "wss://bitpw.m3.ue2.app.chime.aws:443/v2/screen/x",
                        "ScreenSharingUrl": "wss://bitpw.m3.ue2.app.chime.aws:443/v2/screen/x",
                        "ScreenViewingUrl": "wss://bitpw.m3.ue2.app.chime.aws:443/ws/connect",
                        "EventIngestionUrl": "https://data.svc.ue2.ingest.chime.aws/v1/client-events",
                    },
                },
                "attendee": {
                    "AttendeeId": "a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
                    "ExternalUserId": "E012ABCDEF-R09ABCDEF12-U0123456789",
                    "JoinToken": JOIN_TOKEN,
                    "Capabilities": {"Audio": "SendReceive", "Video": "SendReceive"},
                },
            },
        },
        "canvas": {
            "thread_channel_id": "C0123456789",
            "root_thread_ts": "1759000000.123456",
        },
        "huddle": {
            "id": "R09ABCDEF12",
            "created_by": "U0123456789",
            "channel_id": "C0123456789",
            "thread_root_ts": "1759000000.999999",
            "participants": ["U0123456789", "U9876543210"]
            if participants is None
            else participants,
        },
    }


def rooms_info_payload(
    *,
    has_ended: Any = False,
    date_end: Any = 0,
    free_willy: bool = False,
    participants: Any = None,
) -> dict[str, Any]:
    """A ``screenhero.rooms.info`` response, shaped as the doc records it."""
    room: dict[str, Any] = {
        "id": "R09ABCDEF12",
        "call_family": "huddle",
        "created_by": "U0123456789",
        "channels": ["C0123456789"],
        "participants": ["U0123456789"] if participants is None else participants,
        "has_ended": has_ended,
        "date_end": date_end,
        "thread_root_ts": "1759000000.999999",
        "huddle_link": "https://app.slack.com/huddle/E012ABCDEF/C0123456789",
        "media_backend_type": "free_willy",
    }
    if free_willy:
        room["free_willy"] = rooms_join_payload()["call"]["free_willy"]
    return {"ok": True, "room": room}


# -- request shape ----------------------------------------------------------


def test_request_shape_is_minimal_multipart() -> None:
    seen: list[httpx.Request] = []
    client, cfg = client_for(json_handler(rooms_join_payload(), record=seen))

    async def run() -> None:
        async with client:
            await client.rooms_join("C0123456789")

    asyncio.run(run())

    (request,) = seen
    # The workspace host, not slack.com/api.
    assert str(request.url) == "https://hackclub.slack.com/api/rooms.join"
    assert request.method == "POST"
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert request.headers["cookie"] == f"d={XOXD}"
    assert request.headers["user-agent"] == "insomnia/11.2.0"

    fields = parse_multipart(request)
    # hq-fishbowl's minimal shape: exactly these three fields, no `_x_*`,
    # no `multidevice`.
    assert fields == {
        "channel_id": "C0123456789",
        "regions": "us-east-2",
        "token": XOXC,
    }
    assert cfg.media_region == "us-east-2"


def test_explicit_region_overrides_config() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler(rooms_join_payload(), record=seen))

    async def run() -> None:
        async with client:
            await client.rooms_join("D0123456789", regions="ap-southeast-2")

    asyncio.run(run())
    assert parse_multipart(seen[0])["regions"] == "ap-southeast-2"


def test_reason_adds_the_client_mimicking_trio() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True, "room": {}}, record=seen))

    async def run() -> None:
        async with client:
            await client.rooms_info("R09ABCDEF12")

    asyncio.run(run())

    (request,) = seen
    assert str(request.url) == "https://hackclub.slack.com/api/screenhero.rooms.info"
    assert parse_multipart(request) == {
        "room": "R09ABCDEF12",
        "_x_reason": "all-calls-store/conditional-fetch",
        "_x_mode": "online",
        "_x_app_name": "client",
        "token": XOXC,
    }


def test_invite_response_params() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run() -> None:
        async with client:
            await client.rooms_invite_response(
                channel_id="C0123456789", room_id="R09ABCDEF12"
            )

    asyncio.run(run())
    fields = parse_multipart(seen[0])
    assert fields["response"] == "decline"
    assert fields["channel_id"] == "C0123456789"
    assert fields["room_id"] == "R09ABCDEF12"
    assert fields["_x_reason"] == "respond-to-huddle-invite"


def test_extra_headers_are_merged() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(
        json_handler({"ok": True}, record=seen),
        make_config(extra_headers={"X-Extra": "1"}),
    )

    async def run() -> None:
        async with client:
            await client.auth_test()

    asyncio.run(run())
    assert seen[0].headers["x-extra"] == "1"
    assert str(seen[0].url) == "https://hackclub.slack.com/api/auth.test"


def test_params_are_stringified_and_none_dropped() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run() -> None:
        async with client:
            await client.call(
                "channels.prefs.set",
                {"flag": True, "off": False, "count": 3, "skipped": None},
            )

    asyncio.run(run())
    assert parse_multipart(seen[0]) == {
        "flag": "true",
        "off": "false",
        "count": "3",
        "token": XOXC,
    }


def test_injected_client_is_not_closed() -> None:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(json_handler({"ok": True})))
    client = SlackClient(make_config(), client=http_client)

    async def run() -> None:
        async with client:
            await client.auth_test()

    asyncio.run(run())
    assert not http_client.is_closed


# -- rooms.join parsing -----------------------------------------------------


def test_rooms_join_parses_response() -> None:
    client, _ = client_for(json_handler(rooms_join_payload()))

    async def run():
        async with client:
            return await client.rooms_join("C0123456789")

    join = asyncio.run(run())

    assert join.call_id == "R09ABCDEF12"
    assert join.huddle_id == "R09ABCDEF12"
    assert join.created_by == "U0123456789"
    assert join.participants == ("U0123456789", "U9876543210")
    assert join.thread_channel_id == "C0123456789"
    # canvas.root_thread_ts wins over huddle.thread_root_ts.
    assert join.thread_root_ts == "1759000000.123456"

    chime = join.chime
    assert chime.meeting_id == "3f9c1e8a-2b7d-4c5e-9a10-7b6c5d4e3f21"
    assert chime.media_region == "us-east-2"
    assert chime.external_meeting_id == "E012ABCDEF-R09ABCDEF12"
    assert chime.attendee_id == "a1b2c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d"
    assert chime.external_user_id == "E012ABCDEF-R09ABCDEF12-U0123456789"
    assert chime.join_token == JOIN_TOKEN
    assert chime.media_placement.signaling_url.startswith("wss://signal.")
    assert chime.media_placement.audio_host_url == "aud.m3.ue2.app.chime.aws:3478"
    assert chime.media_placement.turn_control_url is not None
    # The raw blob stays available (e.g. Capabilities, AudioFallbackUrl).
    assert chime.raw["attendee"]["Capabilities"]["Audio"] == "SendReceive"
    assert join.raw["call"]["survey_percent"] == 0


def test_participants_as_objects() -> None:
    payload = rooms_join_payload(
        participants=[{"user_id": "U0123456789"}, {"user_id": "U9876543210"}]
    )
    join = parse_rooms_join(payload)
    assert join.participants == ("U0123456789", "U9876543210")


def test_participants_edge_shapes() -> None:
    assert normalise_participants(None) == ()
    assert normalise_participants([]) == ()
    # Mixed/unknown entries are skipped rather than crashing the join.
    assert normalise_participants(
        ["U1", {"user_id": "U2"}, {"nope": 1}, 7, ""]
    ) == ("U1", "U2")


def test_thread_root_ts_falls_back_to_huddle() -> None:
    payload = rooms_join_payload()
    del payload["canvas"]["root_thread_ts"]
    join = parse_rooms_join(payload)
    # hq-fishbowl reads huddle.thread_root_ts.
    assert join.thread_root_ts == "1759000000.999999"


def test_missing_canvas_and_huddle_is_tolerated() -> None:
    payload = rooms_join_payload()
    del payload["canvas"]
    del payload["huddle"]
    join = parse_rooms_join(payload)
    assert join.call_id == "R09ABCDEF12"
    assert join.participants == ()
    assert join.huddle_id is None
    assert join.thread_root_ts is None
    assert join.thread_channel_id is None


def test_missing_free_willy_raises_huddle_error() -> None:
    payload = rooms_join_payload()
    del payload["call"]["free_willy"]
    with pytest.raises(HuddleError, match="free_willy"):
        parse_rooms_join(payload)


def test_missing_join_token_names_the_field() -> None:
    payload = rooms_join_payload()
    del payload["call"]["free_willy"]["attendee"]["JoinToken"]
    with pytest.raises(HuddleError, match="JoinToken"):
        parse_rooms_join(payload)


# -- MeetingFeatures stripping ---------------------------------------------


def test_null_meeting_features_is_stripped() -> None:
    payload = rooms_join_payload(meeting_features=None)
    original = copy.deepcopy(payload)
    join = parse_rooms_join(payload)

    meeting = join.chime.raw["meeting"]
    # The Chime SDK rejects a null MeetingFeatures (huddlefm deletes it).
    assert "MeetingFeatures" not in meeting
    assert meeting["MeetingId"] == original["call"]["free_willy"]["meeting"]["MeetingId"]
    # The response itself is untouched, and still reachable via HuddleJoin.raw.
    assert payload == original
    assert join.raw["call"]["free_willy"]["meeting"]["MeetingFeatures"] is None


def test_non_null_meeting_features_is_kept() -> None:
    features = {"Attendee": {"MaxCount": 250}}
    join = parse_rooms_join(rooms_join_payload(meeting_features=features))
    assert join.chime.raw["meeting"]["MeetingFeatures"] == features


def test_absent_meeting_features_is_fine() -> None:
    payload = rooms_join_payload()
    del payload["call"]["free_willy"]["meeting"]["MeetingFeatures"]
    free_willy = payload["call"]["free_willy"]
    assert strip_null_meeting_features(free_willy) is free_willy


# -- ok: false --------------------------------------------------------------


def test_ok_false_raises_slack_api_error() -> None:
    client, _ = client_for(json_handler({"ok": False, "error": "channel_not_found"}))

    async def run():
        async with client:
            await client.rooms_join("C0000000000")

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    assert info.value.method == "rooms.join"
    assert info.value.error == "channel_not_found"
    assert info.value.response == {"ok": False, "error": "channel_not_found"}
    assert str(info.value) == "rooms.join failed: channel_not_found"


@pytest.mark.parametrize(
    ("code", "expected_in_message"),
    [
        ("team_is_restricted", "enterprise-level"),
        ("not_authed", "Re-extract"),
        ("invalid_auth", "re-extract"),
        ("token_expired", "expired"),
    ],
)
def test_identity_errors_get_an_actionable_hint(code: str, expected_in_message: str) -> None:
    client, _ = client_for(json_handler({"ok": False, "error": code}))

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    message = str(info.value)
    assert code in message
    assert expected_in_message in message
    # The bare code stays machine-readable for callers that branch on it.
    assert info.value.error == code


def test_missing_error_code_still_raises() -> None:
    client, _ = client_for(json_handler({"ok": False}))

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    assert info.value.error == "unknown_error"


def test_rooms_leave_tolerates_feature_not_enabled() -> None:
    payload = {"ok": False, "error": "feature_not_enabled"}
    client, _ = client_for(json_handler(payload))

    async def run():
        async with client:
            return await client.rooms_leave("R09ABCDEF12")

    # The doc: rooms.leave returns feature_not_enabled for this identity, and
    # the Chime LEAVE frame is what actually matters.
    assert asyncio.run(run()) == payload


def test_rooms_leave_still_raises_other_errors() -> None:
    client, _ = client_for(json_handler({"ok": False, "error": "invalid_auth"}))

    async def run():
        async with client:
            await client.rooms_leave("R09ABCDEF12")

    with pytest.raises(SlackApiError):
        asyncio.run(run())


def test_non_json_response_raises_huddle_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>signin</html>", headers={"content-type": "text/html"})

    client, _ = client_for(handler)

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(HuddleError, match="expected JSON"):
        asyncio.run(run())


# -- retries ----------------------------------------------------------------


def test_transport_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huddle_drawings.slack.RETRY_BACKOFF", 0.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"ok": True, "user_id": "U0123456789"})

    client, _ = client_for(handler)

    async def run():
        async with client:
            return await client.auth_test()

    assert asyncio.run(run())["user_id"] == "U0123456789"
    assert calls["n"] == 2


def test_server_errors_are_retried_then_give_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huddle_drawings.slack.RETRY_BACKOFF", 0.0)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    client, _ = client_for(handler)

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(HuddleError, match="giving up"):
        asyncio.run(run())
    assert calls["n"] == 3


def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="not found")

    client, _ = client_for(handler)

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(HuddleError, match="HTTP 404"):
        asyncio.run(run())
    assert calls["n"] == 1


# -- accepting an invite (unverified accept value) ---------------------------


def test_accept_room_invite_returns_the_value_that_worked() -> None:
    seen: list[httpx.Request] = []
    # Only the second candidate "works" -- the first is rejected.
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        value = parse_multipart(request)["response"]
        if value == ACCEPT_INVITE_VALUES[0]:
            return httpx.Response(200, json={"ok": False, "error": "invalid_arguments"})
        return httpx.Response(200, json={"ok": True})

    client, _ = client_for(handler)

    async def run():
        async with client:
            return await client.accept_room_invite(
                channel_id="C0123456789", room_id="R09ABCDEF12"
            )

    assert asyncio.run(run()) == ACCEPT_INVITE_VALUES[1]
    assert len(seen) == 2
    assert all(
        str(r.url) == "https://hackclub.slack.com/api/rooms.inviteResponse" for r in seen
    )
    fields = parse_multipart(seen[-1])
    assert fields["channel_id"] == "C0123456789"
    assert fields["room_id"] == "R09ABCDEF12"
    assert fields["_x_reason"] == "respond-to-huddle-invite"


def test_accept_room_invite_raises_when_every_candidate_fails() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(
        json_handler({"ok": False, "error": "invalid_arguments"}, record=seen)
    )

    async def run():
        async with client:
            await client.accept_room_invite(channel_id="C1", room_id="R1")

    with pytest.raises(HuddleError) as info:
        asyncio.run(run())
    assert len(seen) == len(ACCEPT_INVITE_VALUES)
    message = str(info.value)
    # The message has to name what was tried and point at the live-verified
    # alternative, since the accept value is an open question.
    for value in ACCEPT_INVITE_VALUES:
        assert f"{value}=invalid_arguments" in message
    assert "rooms_join" in message
    assert isinstance(info.value.__cause__, SlackApiError)


def test_accept_room_invite_aborts_on_identity_failure() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": False, "error": "token_expired"}, record=seen))

    async def run():
        async with client:
            await client.accept_room_invite(channel_id="C1", room_id="R1")

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    # A dead session must not be hammered once per candidate.
    assert len(seen) == 1
    assert info.value.error == "token_expired"


def test_accept_room_invite_honours_custom_candidates() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            return await client.accept_room_invite(
                channel_id="C1", room_id="R1", candidates=["yes_please"]
            )

    assert asyncio.run(run()) == "yes_please"
    assert parse_multipart(seen[0])["response"] == "yes_please"


def test_accept_room_invite_rejects_empty_candidates() -> None:
    client, _ = client_for(json_handler({"ok": True}))

    async def run():
        async with client:
            await client.accept_room_invite(channel_id="C1", room_id="R1", candidates=[])

    with pytest.raises(ValueError, match="at least one candidate"):
        asyncio.run(run())


def test_decline_is_still_the_default() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_invite_response(channel_id="C1", room_id="R1")

    asyncio.run(run())
    assert parse_multipart(seen[0])["response"] == "decline"


# -- speculative invite / link wrappers -------------------------------------


def test_send_huddle_invite_shape() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_send_huddle_invite(
                channel_id="C0123456789", room_id="R09ABCDEF12", user_ids=["U1", "U2"]
            )

    asyncio.run(run())
    (request,) = seen
    assert str(request.url) == "https://hackclub.slack.com/api/rooms.sendHuddleInvite"
    assert parse_multipart(request) == {
        "channel_id": "C0123456789",
        "room_id": "R09ABCDEF12",
        "user_ids": "U1,U2",
        "_x_reason": "calls-api/rooms.sendHuddleInvite",
        "_x_mode": "online",
        "_x_app_name": "client",
        "token": XOXC,
    }


def test_user_ids_accepts_a_prejoined_string() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_notify_member(
                channel_id="C1", room_id="R1", user_ids="U1,U2"
            )

    asyncio.run(run())
    assert str(seen[0].url).endswith("/api/rooms.notifyMember")
    assert parse_multipart(seen[0])["user_ids"] == "U1,U2"


def test_empty_user_ids_is_rejected_before_the_request() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_send_huddle_invite(
                channel_id="C1", room_id="R1", user_ids=[]
            )

    with pytest.raises(ValueError, match="user_ids is empty"):
        asyncio.run(run())
    assert seen == []


@pytest.mark.parametrize("method", INVITE_METHOD_CANDIDATES)
def test_rooms_invite_dispatches_to_exactly_one_method(method: str) -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_invite(
                channel_id="C1", room_id="R1", user_ids=["U1"], method=method
            )

    asyncio.run(run())
    # Exactly one call: trying candidates in sequence could double-invite.
    assert len(seen) == 1
    assert str(seen[0].url).endswith(f"/api/{method}")


def test_rooms_invite_defaults_to_send_huddle_invite() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_invite(channel_id="C1", room_id="R1", user_ids=["U1"])

    asyncio.run(run())
    assert str(seen[0].url).endswith("/api/rooms.sendHuddleInvite")


def test_rooms_invite_rejects_unknown_method() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_invite(
                channel_id="C1", room_id="R1", user_ids=["U1"], method="rooms.whatever"
            )

    with pytest.raises(ValueError, match="unknown invite method"):
        asyncio.run(run())
    assert seen == []


def test_rooms_request_omits_missing_room_id() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler({"ok": True}, record=seen))

    async def run():
        async with client:
            await client.rooms_request(channel_id="C1")

    asyncio.run(run())
    fields = parse_multipart(seen[0])
    assert "room_id" not in fields
    assert fields["channel_id"] == "C1"


def test_rooms_get_link_shape() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(
        json_handler({"ok": True, "link": "https://app.slack.com/huddle/E1/C1"}, record=seen)
    )

    async def run():
        async with client:
            return await client.rooms_get_link("R09ABCDEF12")

    payload = asyncio.run(run())
    assert payload["link"].startswith("https://app.slack.com/huddle/")
    assert str(seen[0].url) == "https://hackclub.slack.com/api/rooms.getLink"
    assert parse_multipart(seen[0])["room"] == "R09ABCDEF12"


def test_speculative_methods_surface_slack_errors() -> None:
    client, _ = client_for(json_handler({"ok": False, "error": "invalid_arguments"}))

    async def run():
        async with client:
            await client.rooms_get_link("R1")

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    assert info.value.method == "rooms.getLink"


# -- huddle_info ------------------------------------------------------------


def test_huddle_info_parses_a_live_room() -> None:
    seen: list[httpx.Request] = []
    client, _ = client_for(json_handler(rooms_info_payload(), record=seen))

    async def run():
        async with client:
            return await client.huddle_info("R09ABCDEF12")

    info = asyncio.run(run())

    assert str(seen[0].url) == "https://hackclub.slack.com/api/screenhero.rooms.info"
    assert parse_multipart(seen[0])["room"] == "R09ABCDEF12"
    assert info.room_id == "R09ABCDEF12"
    assert info.has_ended is False
    assert info.is_live is True
    assert info.date_end is None
    assert info.participants == ("U0123456789",)
    assert info.channels == ("C0123456789",)
    assert info.created_by == "U0123456789"
    assert info.huddle_link == "https://app.slack.com/huddle/E012ABCDEF/C0123456789"
    assert info.thread_root_ts == "1759000000.999999"
    assert info.chime is None
    assert info.raw["room"]["media_backend_type"] == "free_willy"


def test_huddle_info_participant_objects() -> None:
    info = parse_room_info(
        rooms_info_payload(participants=[{"user_id": "U1"}, {"user_id": "U2"}])
    )
    assert info.participants == ("U1", "U2")


@pytest.mark.parametrize(
    ("has_ended", "date_end", "expected"),
    [
        (False, 0, False),
        (True, 0, True),
        # Either marker is enough: a stale has_ended=false plus a date_end
        # still means the huddle is over.
        (False, 1759000500, True),
        (True, 1759000500, True),
        (None, None, False),
    ],
)
def test_ended_is_either_marker(has_ended: Any, date_end: Any, expected: bool) -> None:
    info = parse_room_info(rooms_info_payload(has_ended=has_ended, date_end=date_end))
    assert info.has_ended is expected
    assert info.is_live is not expected
    assert info.date_end == (date_end if date_end else None)


def test_huddle_info_extracts_the_chime_blob() -> None:
    info = parse_room_info(rooms_info_payload(free_willy=True))
    assert info.chime is not None
    assert info.chime.meeting_id == "3f9c1e8a-2b7d-4c5e-9a10-7b6c5d4e3f21"
    assert info.chime.join_token == JOIN_TOKEN


def test_huddle_info_strips_null_meeting_features() -> None:
    payload = rooms_info_payload(free_willy=True)
    payload["room"]["free_willy"]["meeting"]["MeetingFeatures"] = None
    info = parse_room_info(payload)
    assert info.chime is not None
    assert "MeetingFeatures" not in info.chime.raw["meeting"]
    # The response is still untouched.
    assert info.raw["room"]["free_willy"]["meeting"]["MeetingFeatures"] is None


def test_huddle_info_tolerates_an_incomplete_chime_blob() -> None:
    payload = rooms_info_payload(free_willy=True)
    del payload["room"]["free_willy"]["attendee"]["JoinToken"]
    info = parse_room_info(payload)
    # rooms.join can still supply the credentials; this must not fail the read.
    assert info.chime is None
    assert info.room_id == "R09ABCDEF12"


def test_huddle_info_missing_room_raises() -> None:
    with pytest.raises(HuddleError, match="no 'room' object"):
        parse_room_info({"ok": True})


def test_huddle_info_tolerates_sparse_rooms() -> None:
    info = parse_room_info({"ok": True, "room": {}})
    assert info.room_id is None
    assert info.has_ended is False
    assert info.participants == ()
    assert info.channels == ()
    assert info.huddle_link is None


def test_huddle_info_canvas_thread_ts_fallback() -> None:
    payload = rooms_info_payload()
    del payload["room"]["thread_root_ts"]
    payload["room"]["canvas_thread_ts"] = "1759000111.000100"
    assert parse_room_info(payload).thread_root_ts == "1759000111.000100"


# -- secrets never leak -----------------------------------------------------


def _assert_no_secrets(text: str) -> None:
    for secret in (XOXC, XOXD, JOIN_TOKEN):
        assert secret not in text, f"secret leaked: {text!r}"


def test_slack_api_error_text_has_no_tokens() -> None:
    client, _ = client_for(
        json_handler(
            {
                "ok": False,
                "error": "team_is_restricted",
                # Even if Slack echoed something sensitive back:
                "echo": {"token": XOXC, "JoinToken": JOIN_TOKEN},
            }
        )
    )

    async def run():
        async with client:
            await client.rooms_join("C0123456789")

    with pytest.raises(SlackApiError) as info:
        asyncio.run(run())
    _assert_no_secrets(str(info.value))
    _assert_no_secrets(repr(info.value))
    _assert_no_secrets("".join(str(a) for a in info.value.args))


def test_http_error_body_is_scrubbed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # A hostile/sloppy error page echoing the credentials back.
        return httpx.Response(400, text=f"bad request: token={XOXC} cookie=d={XOXD}")

    client, _ = client_for(handler)

    async def run():
        async with client:
            await client.auth_test()

    with pytest.raises(HuddleError) as info:
        asyncio.run(run())
    _assert_no_secrets(str(info.value))
    _assert_no_secrets(repr(info.value))
    assert "<redacted>" in str(info.value)


def test_chime_credentials_repr_hides_join_token() -> None:
    join = parse_rooms_join(rooms_join_payload())
    _assert_no_secrets(repr(join.chime))
    _assert_no_secrets(repr(join))


def test_new_method_paths_leak_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """The invite/link/info additions must stay token-free too."""
    caplog.set_level("DEBUG", logger="huddle_drawings.slack")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("screenhero.rooms.info"):
            return httpx.Response(200, json=rooms_info_payload(free_willy=True))
        if request.url.path.endswith("rooms.inviteResponse"):
            return httpx.Response(200, json={"ok": False, "error": "invalid_arguments"})
        return httpx.Response(200, json={"ok": True, "link": "https://app.slack.com/huddle/E1/C1"})

    client, _ = client_for(handler)

    async def run():
        async with client:
            info = await client.huddle_info("R09ABCDEF12")
            await client.rooms_get_link("R09ABCDEF12")
            await client.rooms_send_huddle_invite(
                channel_id="C0123456789", room_id="R09ABCDEF12", user_ids=["U1"]
            )
            with pytest.raises(HuddleError) as err:
                await client.accept_room_invite(
                    channel_id="C0123456789", room_id="R09ABCDEF12"
                )
            return info, err.value

    info, err = asyncio.run(run())

    assert info.chime is not None and info.chime.join_token == JOIN_TOKEN
    _assert_no_secrets(repr(info))
    _assert_no_secrets(str(err))
    _assert_no_secrets(repr(err))
    _assert_no_secrets(caplog.text)
    # The token is in the bodies we sent, so it must not be in the logs.
    assert any(XOXC.encode() in r.content for r in seen)


def test_nothing_sensitive_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG", logger="huddle_drawings.slack")
    client, _ = client_for(json_handler({"ok": False, "error": "token_expired"}))

    async def run():
        async with client:
            await client.rooms_join("C0123456789")

    with pytest.raises(SlackApiError):
        asyncio.run(run())
    _assert_no_secrets(caplog.text)
    assert "rooms.join failed: token_expired" in caplog.text


# --------------------------------------------------------------------------- #
# user id -> DM channel resolution
# --------------------------------------------------------------------------- #


def _client_with(handler):
    """A SlackClient wired to a MockTransport handler."""
    import httpx

    from huddle_drawings.config import Config

    config = Config(
        xoxc_token="xoxc-aaaaaaaaaa",
        xoxd_token="xoxd-bbbbbbbbbb",
        slack_host="example.slack.com",
    )
    return SlackClient(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


@pytest.mark.parametrize("target", ["C0123456789", "D0123456789", "G0123456789"])
def test_resolve_passes_conversation_ids_through(target: str) -> None:
    import httpx

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"ok": True})

    client = _client_with(handler)

    async def go() -> str:
        try:
            return await client.resolve_conversation_id(target)
        finally:
            await client.close()

    assert asyncio.run(go()) == target
    assert calls == [], "a conversation id must not cost an API call"


@pytest.mark.parametrize("user", ["U0123456789", "W0123456789"])
def test_resolve_opens_a_dm_for_a_user_id(user: str) -> None:
    import httpx

    seen: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.content))
        return httpx.Response(
            200, json={"ok": True, "channel": {"id": "D9998887776"}}
        )

    client = _client_with(handler)

    async def go() -> str:
        try:
            return await client.resolve_conversation_id(user)
        finally:
            await client.close()

    assert asyncio.run(go()) == "D9998887776"
    assert seen[0][0] == "/api/conversations.open"
    assert user.encode() in seen[0][1]


def test_resolve_falls_back_to_im_open() -> None:
    """conversations.* can be blocked for a browser-session identity on Grid."""
    import httpx

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("conversations.open"):
            return httpx.Response(200, json={"ok": False, "error": "missing_scope"})
        return httpx.Response(
            200, json={"ok": True, "channel": {"id": "D5554443332"}}
        )

    client = _client_with(handler)

    async def go() -> str:
        try:
            return await client.resolve_conversation_id("U0123456789")
        finally:
            await client.close()

    assert asyncio.run(go()) == "D5554443332"
    assert paths == ["/api/conversations.open", "/api/im.open"]


def test_resolve_raises_helpfully_when_no_dm_can_be_opened() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "user_not_found"})

    client = _client_with(handler)

    async def go() -> None:
        try:
            await client.resolve_conversation_id("U0123456789")
        finally:
            await client.close()

    with pytest.raises(HuddleError, match="could not open a DM"):
        asyncio.run(go())


def test_resolve_does_not_retry_a_dead_session() -> None:
    """A fatal identity error must abort, not try the fallback method too."""
    import httpx

    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

    client = _client_with(handler)

    async def go() -> None:
        try:
            await client.resolve_conversation_id("U0123456789")
        finally:
            await client.close()

    with pytest.raises(SlackApiError):
        asyncio.run(go())
    assert paths == ["/api/conversations.open"]


def test_resolve_rejects_an_empty_target() -> None:
    client = _client_with(lambda r: None)

    async def go() -> None:
        try:
            await client.resolve_conversation_id("   ")
        finally:
            await client.close()

    with pytest.raises(HuddleError, match="empty"):
        asyncio.run(go())


def test_resolve_passes_unknown_prefixes_through_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _client_with(lambda r: None)

    async def go() -> str:
        try:
            return await client.resolve_conversation_id("X123")
        finally:
            await client.close()

    with caplog.at_level("WARNING"):
        assert asyncio.run(go()) == "X123"
    assert "does not look like" in caplog.text
