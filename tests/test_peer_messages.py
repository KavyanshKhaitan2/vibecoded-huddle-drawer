"""Builder/parser round-trips for every ``PeerMessage`` payload field.

No network, no asyncio: every test builds bytes and decodes them with
:func:`peer_messages.parse_peer_message`, then asserts on the protobuf.
"""

from __future__ import annotations

import pytest

from huddle_drawings import peer_messages as pm
from huddle_drawings.config import MAX_DATA_MESSAGE_BYTES
from huddle_drawings.proto import peer_message_pb2 as pb

SENDER = "att-sender-1"
SHARER = "att-sharer-9"


def parse(data: bytes) -> pb.PeerMessage:
    return pm.parse_peer_message(data)


# --------------------------------------------------------------------------
# every builder sets fromPeerIdString
# --------------------------------------------------------------------------

ALL_BUILDERS = [
    ("draw", lambda: pm.build_draw(
        from_peer_id=SENDER, draw_type=pm.DrawType.END, item_id=SHARER)),
    ("reacji", lambda: pm.build_reacji(from_peer_id=SENDER, emoji="tada")),
    ("capabilities", lambda: pm.build_capabilities(
        from_peer_id=SENDER, is_draw_enabled=True)),
    ("background", lambda: pm.build_background(
        from_peer_id=SENDER, url="https://example.invalid/bg.png")),
    ("muteRequest", lambda: pm.build_mute_request(
        from_peer_id=SENDER, peer_id="att-victim")),
    ("awarenessBroadcast", lambda: pm.build_awareness_broadcast(
        from_peer_id=SENDER, channel="C1", user="U1", state={"a": "b"})),
    ("awarenessQuery", lambda: pm.build_awareness_query(
        from_peer_id=SENDER, channel="C1", user="U1")),
    ("awarenessClose", lambda: pm.build_awareness_close(
        from_peer_id=SENDER, channel="C1", user="U1")),
    ("jukeboxRequest", lambda: pm.build_jukebox_request(
        from_peer_id=SENDER, song_key="song-1")),
    ("contentShareDetails", lambda: pm.build_content_share_details(
        from_peer_id=SENDER,
        source_type=pm.ContentShareSource.DESKTOP_SCREEN,
        # Normally the sender's own id; a literal here so that blanking SENDER
        # in test_sender_is_required leaves only from_peer_id missing.
        item_id="att-some-sharer")),
    ("recordingStartNotification", lambda: pm.build_recording_start_notification(
        from_peer_id=SENDER)),
    ("transcriptCatchUp", lambda: pm.build_transcript_solicit(from_peer_id=SENDER)),
    ("kickNotification", lambda: pm.build_kick_notification(
        from_peer_id=SENDER, kicked_user_id="U123")),
]


@pytest.mark.parametrize(("field", "build"), ALL_BUILDERS, ids=[b[0] for b in ALL_BUILDERS])
def test_every_builder_round_trips_and_sets_sender(field: str, build) -> None:
    data = build()
    msg = parse(data)
    # The doc: "Every outgoing message sets fromPeerIdString to the sender's
    # Chime attendeeId". The legacy numeric field stays unset.
    assert msg.fromPeerIdString == SENDER
    assert msg.fromPeerId == 0
    assert msg.HasField(field)
    # Exactly one payload field per message.
    present = [name for name in pm.PAYLOAD_FIELDS if msg.HasField(name)]
    assert present == [field]
    assert len(data) <= MAX_DATA_MESSAGE_BYTES
    assert describe_mentions(data, field)


def describe_mentions(data: bytes, field: str) -> bool:
    text = pm.describe_peer_message(data)
    return SENDER in text and field.lower()[:5] in text.lower()


def test_every_schema_payload_field_has_a_builder() -> None:
    """Guards against the schema growing a field with no builder."""
    schema_fields = {
        f.name
        for f in pb.PeerMessage.DESCRIPTOR.fields
        if f.name not in ("fromPeerId", "fromPeerIdString")
    }
    assert schema_fields == set(pm.PAYLOAD_FIELDS)
    assert {name for name, _ in ALL_BUILDERS} == schema_fields


@pytest.mark.parametrize(("field", "build"), ALL_BUILDERS, ids=[b[0] for b in ALL_BUILDERS])
def test_sender_is_required(field: str, build) -> None:
    global SENDER
    original, SENDER = SENDER, ""
    try:
        with pytest.raises(ValueError, match="from_peer_id"):
            build()
    finally:
        SENDER = original


# --------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------


def test_draw_carries_segment_item_id_and_type() -> None:
    seg = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)]
    msg = parse(
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.NEW,
            item_id=SHARER,
            segment=seg,
        )
    )
    assert msg.draw.type == pm.DrawType.NEW
    assert msg.draw.itemId == SHARER
    assert (msg.draw.segment.a.x, msg.draw.segment.a.y) == (0.1, 0.2)
    assert (msg.draw.segment.d.x, msg.draw.segment.d.y) == (0.7, 0.8)


def test_draw_is_about_a_hundred_bytes() -> None:
    """The doc: "a draw message is ~100 bytes"."""
    data = pm.build_draw(
        from_peer_id=SENDER,
        draw_type=pm.DrawType.NEW,
        item_id=SHARER,
        segment=[(0.1, 0.2)] * 4,
    )
    assert 60 <= len(data) <= 140


def test_draw_requires_item_id() -> None:
    with pytest.raises(ValueError, match="item_id"):
        pm.build_draw(from_peer_id=SENDER, draw_type=pm.DrawType.END, item_id="")


def test_draw_accepts_geometry_segments() -> None:
    from huddle_drawings.geometry import Point, Segment

    seg = Segment.degenerate(Point(0.25, 0.75))
    msg = parse(
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.NEW,
            item_id=SHARER,
            segment=seg,
        )
    )
    for name in "abcd":
        point = getattr(msg.draw.segment, name)
        assert (point.x, point.y) == (0.25, 0.75)


def test_draw_position_carries_x_y() -> None:
    msg = parse(
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.POSITION,
            item_id=SHARER,
            x=-5.0,
            y=-5.0,
        )
    )
    assert msg.draw.type == pm.DrawType.POSITION
    assert (msg.draw.x, msg.draw.y) == (-5.0, -5.0)
    assert not msg.draw.HasField("segment")


@pytest.mark.parametrize("hue", [0, 1, 180, 359])
def test_valid_hues(hue: int) -> None:
    assert pm.validate_hue(hue) == hue
    msg = parse(
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.CHANGE_HUE,
            item_id=SHARER,
            hue=hue,
        )
    )
    assert msg.draw.hue == hue


@pytest.mark.parametrize("hue", [-1, 360, 361, 1000, -180])
def test_out_of_range_hues_are_rejected(hue: int) -> None:
    with pytest.raises(ValueError, match="0..359"):
        pm.validate_hue(hue)
    with pytest.raises(ValueError):
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.CHANGE_HUE,
            item_id=SHARER,
            hue=hue,
        )


@pytest.mark.parametrize("hue", [1.5, "180", None, True])
def test_non_integer_hues_are_rejected(hue: object) -> None:
    with pytest.raises(TypeError):
        pm.validate_hue(hue)  # type: ignore[arg-type]


def test_segment_needs_four_points() -> None:
    with pytest.raises(ValueError, match="4 points"):
        pm.build_draw(
            from_peer_id=SENDER,
            draw_type=pm.DrawType.NEW,
            item_id=SHARER,
            segment=[(0.1, 0.1), (0.2, 0.2)],
        )


# --------------------------------------------------------------------------
# content share / capabilities
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "desktop:screen",
        "desktop:window",
        "desktop:unknown",
        "browser:content",
        "mobile:screen",
        "unknown",
    ],
)
def test_content_share_source_values_from_the_doc(value: str) -> None:
    msg = parse(
        pm.build_content_share_details(
            from_peer_id=SENDER, source_type=value, item_id=SENDER
        )
    )
    assert msg.contentShareDetails.sourceType == value
    assert msg.contentShareDetails.itemId == SENDER


def test_content_share_source_enum_is_exactly_the_documented_set() -> None:
    assert {s.value for s in pm.ContentShareSource} == {
        "desktop:screen",
        "desktop:window",
        "desktop:unknown",
        "browser:content",
        "mobile:screen",
        "unknown",
    }
    # Drawing is allowed only for screen, plus window on new enough desktop.
    assert pm.DRAWABLE_SOURCE_TYPES == {
        pm.ContentShareSource.DESKTOP_SCREEN,
        pm.ContentShareSource.DESKTOP_WINDOW,
    }


@pytest.mark.parametrize("value", ["desktop", "screen", "DESKTOP:SCREEN", "x11:screen", ""])
def test_unknown_source_type_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        pm.build_content_share_details(
            from_peer_id=SENDER, source_type=value, item_id=SENDER
        )


def test_content_share_requires_item_id() -> None:
    with pytest.raises(ValueError, match="item_id"):
        pm.build_content_share_details(
            from_peer_id=SENDER,
            source_type=pm.ContentShareSource.DESKTOP_SCREEN,
            item_id="",
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_capabilities_round_trip(enabled: bool) -> None:
    msg = parse(pm.build_capabilities(from_peer_id=SENDER, is_draw_enabled=enabled))
    assert msg.HasField("capabilities")
    assert msg.capabilities.isDrawEnabled is enabled


# --------------------------------------------------------------------------
# reacji / jukebox / kick / mute
# --------------------------------------------------------------------------


def test_reacji_strips_colons_and_defaults_to_standard() -> None:
    msg = parse(pm.build_reacji(from_peer_id=SENDER, emoji=":tada:"))
    assert msg.reacji.emoji == "tada"
    assert msg.reacji.type == pm.ReacjiType.Standard
    assert msg.reacji.emojiUrl == ""


def test_reacji_with_url_and_type() -> None:
    msg = parse(
        pm.build_reacji(
            from_peer_id=SENDER,
            emoji="party",
            reacji_type=pm.ReacjiType.Gif,
            emoji_url="https://example.invalid/p.gif",
        )
    )
    assert msg.reacji.type == pm.ReacjiType.Gif
    assert msg.reacji.emojiUrl.endswith("p.gif")


def test_reacji_requires_emoji() -> None:
    with pytest.raises(ValueError, match="emoji"):
        pm.build_reacji(from_peer_id=SENDER, emoji="")


def test_jukebox_play_and_stop() -> None:
    play = parse(pm.build_jukebox_request(from_peer_id=SENDER, song_key="s1"))
    assert play.jukeboxRequest.songKey == "s1"
    assert play.jukeboxRequest.stop is False
    stop = parse(pm.build_jukebox_request(from_peer_id=SENDER, stop=True))
    assert stop.jukeboxRequest.stop is True
    with pytest.raises(ValueError, match="song_key"):
        pm.build_jukebox_request(from_peer_id=SENDER)


def test_mute_request_targets_an_attendee() -> None:
    msg = parse(pm.build_mute_request(from_peer_id=SENDER, peer_id="att-victim"))
    assert msg.muteRequest.peerId == "att-victim"
    with pytest.raises(ValueError, match="peer_id"):
        pm.build_mute_request(from_peer_id=SENDER, peer_id="")


def test_kick_notification_targets_a_user_id() -> None:
    msg = parse(pm.build_kick_notification(from_peer_id=SENDER, kicked_user_id="U42"))
    assert msg.kickNotification.kickedUserId == "U42"
    with pytest.raises(ValueError, match="kicked_user_id"):
        pm.build_kick_notification(from_peer_id=SENDER, kicked_user_id="")


def test_safety_note_is_documented_on_the_two_dangerous_builders() -> None:
    """The doc's safety note must be visible to whoever calls these."""
    for builder in (pm.build_mute_request, pm.build_kick_notification):
        doc = (builder.__doc__ or "").lower()
        assert "safety" in doc
        assert "have not agreed" in doc or "haven't agreed" in doc


def test_no_receive_branch_is_documented() -> None:
    """background and recordingStartNotification are sent and ignored."""
    for builder in (pm.build_background, pm.build_recording_start_notification):
        doc = (builder.__doc__ or "").lower()
        assert "no receive branch" in doc


# --------------------------------------------------------------------------
# background / recording
# --------------------------------------------------------------------------


def test_background_round_trip() -> None:
    msg = parse(pm.build_background(from_peer_id=SENDER, url="x://y"))
    assert msg.background.url == "x://y"
    assert "no receiver" in pm.describe_peer_message(msg)


def test_recording_start_notification_round_trip() -> None:
    msg = parse(
        pm.build_recording_start_notification(
            from_peer_id=SENDER, recording_type="audio"
        )
    )
    assert msg.recordingStartNotification.type == pm.RecordingType.SUMMARY
    assert msg.recordingStartNotification.recordingType == "audio"
    assert "no receiver" in pm.describe_peer_message(msg)


# --------------------------------------------------------------------------
# awareness
# --------------------------------------------------------------------------


def test_awareness_broadcast_accepts_mapping_and_fields() -> None:
    msg = parse(
        pm.build_awareness_broadcast(
            from_peer_id=SENDER,
            channel="C1",
            user="U1",
            state=[
                pm.AwarenessField("typing", "true", 1700000000000),
                ("mood", "ok"),
            ],
        )
    )
    state = msg.awarenessBroadcast.awarenessState
    assert [(p.field_name, p.field_value) for p in state] == [
        ("typing", "true"),
        ("mood", "ok"),
    ]
    # field_timestamp_ms is `optional`, so absence is observable.
    assert state[0].HasField("field_timestamp_ms")
    assert state[0].field_timestamp_ms == 1700000000000
    assert not state[1].HasField("field_timestamp_ms")


def test_awareness_query_and_close() -> None:
    q = parse(pm.build_awareness_query(from_peer_id=SENDER, channel="C", user="U"))
    assert (q.awarenessQuery.channel, q.awarenessQuery.user) == ("C", "U")
    c = parse(pm.build_awareness_close(from_peer_id=SENDER, channel="C", user="U"))
    assert (c.awarenessClose.channel, c.awarenessClose.user) == ("C", "U")


# --------------------------------------------------------------------------
# transcript catch-up
# --------------------------------------------------------------------------


def test_request_id_shapes_match_the_doc() -> None:
    assert pm.make_request_id("att1", now_ms=5) == "att1-5"
    assert pm.make_request_id("att1", probe=True, now_ms=5) == "att1-probe-5"
    assert pm.make_request_from_id("att2", "att1-5") == "att2:att1-5"


def test_solicit_offer_request_from() -> None:
    solicit = parse(pm.build_transcript_solicit(from_peer_id=SENDER))
    assert solicit.transcriptCatchUp.type == pm.TranscriptType.SOLICIT
    assert solicit.transcriptCatchUp.request_id.startswith(f"{SENDER}-")

    offer = parse(
        pm.build_transcript_offer(
            from_peer_id=SENDER,
            request_id="att1-5",
            earliest_transcript_timestamp_ms=1234,
        )
    )
    assert offer.transcriptCatchUp.type == pm.TranscriptType.OFFER
    assert offer.transcriptCatchUp.earliest_transcript_timestamp_ms == 1234

    req = parse(
        pm.build_transcript_request_from(
            from_peer_id=SENDER, chosen_attendee_id="att2", request_id="att1-5"
        )
    )
    assert req.transcriptCatchUp.type == pm.TranscriptType.REQUEST_FROM
    assert req.transcriptCatchUp.request_id == "att2:att1-5"


def test_small_transcript_is_one_response() -> None:
    lines = [pm.TranscriptLine("U1", "hello", 1, 2)]
    out = pm.build_transcript_response(
        from_peer_id=SENDER, request_id="r1", lines=lines
    )
    assert len(out) == 1
    msg = parse(out[0])
    assert msg.transcriptCatchUp.type == pm.TranscriptType.RESPONSE
    assert len(out[0]) < pm.TRANSCRIPT_RESPONSE_MAX_BYTES
    assert msg.transcriptCatchUp.lines[0].content == "hello"
    assert msg.transcriptCatchUp.total_chunks == 0


def test_large_transcript_is_chunked_under_2032_bytes() -> None:
    lines = [
        pm.TranscriptLine("U%d" % i, "a transcript line that says words " * 5, i, i + 1)
        for i in range(40)
    ]
    out = pm.build_transcript_response(
        from_peer_id=SENDER, request_id="r1", lines=lines
    )
    assert len(out) > 1
    seen: list[str] = []
    for index, data in enumerate(out):
        assert len(data) < pm.TRANSCRIPT_CHUNK_MAX_BYTES
        msg = parse(data)
        catch_up = msg.transcriptCatchUp
        assert catch_up.type == pm.TranscriptType.RESPONSE_CHUNK
        assert catch_up.chunk_index == index
        assert catch_up.total_chunks == len(out)
        assert catch_up.request_id == "r1"
        seen.extend(line.content for line in catch_up.lines)
    assert seen == [line.content for line in lines]


def test_one_overlong_line_is_split_across_chunks() -> None:
    long_line = pm.TranscriptLine("U1", "x" * 6000, 10, 20)
    out = pm.build_transcript_response(
        from_peer_id=SENDER, request_id="r1", lines=[long_line]
    )
    assert len(out) >= 3
    rebuilt = ""
    for data in out:
        assert len(data) < pm.TRANSCRIPT_CHUNK_MAX_BYTES
        for line in parse(data).transcriptCatchUp.lines:
            assert line.user_id == "U1"
            assert (line.start_time_ms, line.end_time_ms) == (10, 20)
            rebuilt += line.content
    assert rebuilt == long_line.content


def test_chunking_never_splits_a_code_point() -> None:
    out = pm.build_transcript_response(
        from_peer_id=SENDER,
        request_id="r1",
        lines=[pm.TranscriptLine("U1", "é" * 4000, 0, 0)],
    )
    rebuilt = "".join(
        line.content for data in out for line in parse(data).transcriptCatchUp.lines
    )
    assert rebuilt == "é" * 4000


def test_split_transcript_lines_groups_lines() -> None:
    lines = [pm.TranscriptLine("U1", "word " * 20, 0, 1) for _ in range(30)]
    groups = pm.split_transcript_lines(
        from_peer_id=SENDER, request_id="r1", lines=lines
    )
    assert sum(len(g) for g in groups) == len(lines)
    assert all(groups)


def test_protobuf_transcript_lines_are_accepted() -> None:
    raw = pb.PeerMessage.TranscriptLine(
        user_id="U1", content="hi", start_time_ms=1, end_time_ms=2
    )
    msg = parse(
        pm.build_transcript_catch_up(
            from_peer_id=SENDER,
            catch_up_type=pm.TranscriptType.RESPONSE,
            request_id="r",
            lines=[raw],
        )
    )
    assert msg.transcriptCatchUp.lines[0].content == "hi"


# --------------------------------------------------------------------------
# size limit / parsing / describing
# --------------------------------------------------------------------------


def test_oversize_payload_raises_instead_of_vanishing() -> None:
    with pytest.raises(pm.PayloadTooLarge) as excinfo:
        pm.build_background(from_peer_id=SENDER, url="u" * 4000)
    assert excinfo.value.limit == MAX_DATA_MESSAGE_BYTES
    assert excinfo.value.size > MAX_DATA_MESSAGE_BYTES


def test_payload_at_the_limit_is_allowed() -> None:
    data = pm.build_background(from_peer_id=SENDER, url="u" * 1900)
    assert len(data) <= MAX_DATA_MESSAGE_BYTES


def test_parse_peer_message_accepts_bytearray() -> None:
    data = pm.build_awareness_query(from_peer_id=SENDER, channel="C", user="U")
    assert pm.parse_peer_message(bytearray(data)).awarenessQuery.channel == "C"


def test_describe_degenerate_segment() -> None:
    data = pm.build_draw(
        from_peer_id=SENDER,
        draw_type=pm.DrawType.NEW,
        item_id=SHARER,
        segment=[(0.5, 0.5)] * 4,
    )
    text = pm.describe_peer_message(data)
    assert "NEW" in text
    assert "degenerate" in text
    assert SHARER in text


def test_describe_empty_message() -> None:
    msg = pb.PeerMessage(fromPeerIdString=SENDER)
    assert pm.describe_peer_message(msg) == f"PeerMessage(from={SENDER}, empty)"
