"""Builders and parsers for ``PeerMessage`` -- the huddle data-channel payload.

Every payload that travels inside a huddle's Chime session is a
``slack.calls.client.PeerMessage`` (binary protobuf), sent as a Chime data
message on one of three topics (``docs/huddles-peer-messages.md#transport``):

* ``data-channel`` -- draw, reacji, capabilities, background, muteRequest,
  jukeboxRequest, contentShareDetails, recordingStartNotification,
  kickNotification
* ``awareness-sync-channel`` -- awarenessBroadcast / awarenessQuery /
  awarenessClose
* ``transcript-catchup`` -- transcriptCatchUp

There is one builder per payload field of the schema. They all return ``bytes``
ready to hand to ``send_data_message``, and they all set ``fromPeerIdString``
to the sender's Chime ``attendeeId`` -- the doc: "Every outgoing message sets
``fromPeerIdString`` to the sender's Chime ``attendeeId``". The legacy numeric
``fromPeerId`` (field 3) is never set by the real client, so it is never set
here either.

Receive-side notes that apply to *everything* here
(``docs/huddles-peer-messages.md``):

* These are fire-and-forget broadcasts to every attendee in the meeting. There
  is no request/response pairing except the transcript catch-up and awareness
  handshakes.
* Receivers ignore messages from their own attendee id.
* ``draw``, ``reacji``, ``muteRequest`` and ``jukeboxRequest`` are dropped when
  the sender is not in the receiver's peer cache ("Dropping draw message from
  unknown peer"). A sender only enters that cache once Chime has reported it in
  ``AUDIO_STREAM_ID_INFO``, which needs the attendee's **audio connection** to
  be up -- a signaling-only connection can send, but nothing it sends is acted
  on.
* Chime drops any payload over :data:`config.MAX_DATA_MESSAGE_BYTES`, so every
  builder here checks the serialized size and raises :class:`PayloadTooLarge`
  rather than letting something vanish on the wire.
* The per-meeting rate limit (~200 burst, ~150/s, shared by all senders) is
  enforced one layer down, in ``ratelimit.TokenBucket``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .config import MAX_DATA_MESSAGE_BYTES
from .proto import peer_message_pb2 as pb
from .types import HuddleError

logger = logging.getLogger(__name__)

PeerMessage: TypeAlias = pb.PeerMessage
ProtoPoint: TypeAlias = pb.PeerMessage.Point
ProtoSegment: TypeAlias = pb.PeerMessage.Segment

#: Enum aliases, so callers need not reach into the generated module.
DrawType = pb.PeerMessage.Draw.Type
ReacjiType = pb.PeerMessage.Reacji.Type
RecordingType = pb.PeerMessage.RecordingStartNotification.Type
TranscriptType = pb.PeerMessage.TranscriptCatchUp.Type

__all__ = [
    "DRAWABLE_SOURCE_TYPES",
    "HUE_MAX",
    "HUE_MIN",
    "PAYLOAD_FIELDS",
    "TRANSCRIPT_CHUNK_MAX_BYTES",
    "TRANSCRIPT_RESPONSE_MAX_BYTES",
    "AwarenessField",
    "ContentShareSource",
    "DrawType",
    "PayloadTooLarge",
    "PeerMessage",
    "ProtoPoint",
    "ProtoSegment",
    "ReacjiType",
    "RecordingType",
    "TranscriptLine",
    "TranscriptType",
    "build_awareness_broadcast",
    "build_awareness_close",
    "build_awareness_query",
    "build_background",
    "build_capabilities",
    "build_content_share_details",
    "build_draw",
    "build_jukebox_request",
    "build_kick_notification",
    "build_mute_request",
    "build_reacji",
    "build_recording_start_notification",
    "build_transcript_catch_up",
    "build_transcript_offer",
    "build_transcript_request_from",
    "build_transcript_response",
    "build_transcript_solicit",
    "coerce_segment",
    "describe_peer_message",
    "make_request_from_id",
    "make_request_id",
    "parse_peer_message",
    "split_transcript_lines",
    "validate_hue",
]


class PayloadTooLarge(HuddleError):
    """A serialized ``PeerMessage`` is over the size Chime will carry.

    Chime drops oversize payloads silently, so this is raised at build time
    instead. ``docs/huddles-peer-messages.md``: "Stay under 2048 bytes per
    message; a draw message is ~100 bytes."
    """

    def __init__(self, size: int, limit: int, what: str = "PeerMessage") -> None:
        super().__init__(f"{what} is {size} bytes, over the {limit}-byte limit")
        self.size = size
        self.limit = limit


# --------------------------------------------------------------------------
# small value types / coercion helpers
# --------------------------------------------------------------------------

PointLike: TypeAlias = "ProtoPoint | tuple[float, float] | Sequence[float] | object"
SegmentLike: TypeAlias = "ProtoSegment | Sequence[PointLike] | object"


def _point_xy(value: PointLike) -> tuple[float, float]:
    """Coerce anything point-shaped to ``(x, y)`` floats."""
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)  # type: ignore[attr-defined]
    try:
        px, py = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"not a point: {value!r}") from exc
    return float(px), float(py)


def _fill_point(target: ProtoPoint, value: PointLike) -> None:
    target.x, target.y = _point_xy(value)


def _coerce_segment(value: SegmentLike) -> ProtoSegment:
    """Coerce a cubic Bézier to the protobuf ``Segment``.

    Accepts a protobuf ``Segment``, anything with ``a``/``b``/``c``/``d``
    attributes (e.g. :class:`geometry.Segment`), or a 4-sequence of points.
    The coercion lives here so that ``geometry`` stays an optional dependency
    for callers who only want to send raw numbers.
    """
    if isinstance(value, ProtoSegment):
        return value
    out = ProtoSegment()
    if all(hasattr(value, name) for name in "abcd"):
        for name in "abcd":
            _fill_point(getattr(out, name), getattr(value, name))
        return out
    pts = list(value)  # type: ignore[call-overload]
    if len(pts) != 4:
        raise ValueError(f"a cubic Bézier segment needs 4 points, got {len(pts)}")
    for name, pt in zip("abcd", pts, strict=True):
        _fill_point(getattr(out, name), pt)
    return out


#: Public name for the segment coercion above, for other modules.
coerce_segment = _coerce_segment


def _message(from_peer_id: str) -> PeerMessage:
    if not from_peer_id:
        raise ValueError(
            "from_peer_id is required: every outgoing PeerMessage sets "
            "fromPeerIdString to the sender's Chime attendeeId"
        )
    msg = PeerMessage()
    msg.fromPeerIdString = from_peer_id
    return msg


def _serialize(
    msg: PeerMessage,
    *,
    limit: int = MAX_DATA_MESSAGE_BYTES,
    what: str = "PeerMessage",
) -> bytes:
    data = msg.SerializeToString()
    if len(data) > limit:
        raise PayloadTooLarge(len(data), limit, what)
    return data


# --------------------------------------------------------------------------
# draw (field 2)
# --------------------------------------------------------------------------

HUE_MIN = 0
HUE_MAX = 359


def validate_hue(hue: int) -> int:
    """Check ``hue`` is a usable HSL hue, in degrees.

    ``CHANGE_HUE`` carries ``hue`` as an int32 in **degrees 0-359**. The
    renderer always draws ``hsla(hue, 100%, 75%)`` with an
    ``hsla(hue, 80%, 50%)`` shadow, so saturation, lightness, the 7 px width and
    the round caps are all fixed. Only the hue is controllable: every colour is
    a pastel, and **nothing can be drawn dark, grey or white**. One attendee
    owns exactly one line in one hue, so N colours at once needs N accounts.
    """
    if isinstance(hue, bool) or not isinstance(hue, int):
        raise TypeError(f"hue must be an int in degrees, got {hue!r}")
    if not HUE_MIN <= hue <= HUE_MAX:
        raise ValueError(f"hue must be {HUE_MIN}..{HUE_MAX} degrees, got {hue}")
    return hue


def build_draw(
    *,
    from_peer_id: str,
    draw_type: DrawType.ValueType,
    item_id: str,
    segment: SegmentLike | None = None,
    hue: int | None = None,
    x: float | None = None,
    y: float | None = None,
) -> bytes:
    """Build ``PeerMessage.draw`` -- one cubic Bézier (or cursor move) for a share.

    Receivers publish ``{id: senderAttendeeId, ...draw}`` to the draw renderer,
    which keeps **one line per sender attendee** in one hue.

    ``item_id`` **must be the sharer's attendeeId**: the share is listed in
    Chime's ``INDEX`` as ``<sharerAttendeeId>#content``, and the ``itemId`` to
    use is that with ``#content`` stripped
    (:attr:`types.StreamSource.sharer_attendee_id`). The renderer drops draws
    whose ``itemId`` does not match the share it is drawing over, so a wrong
    ``itemId`` means the drawing silently never appears. A share's attendeeId
    changes every time the share restarts, so long-running drawers must follow
    ``INDEX`` updates.

    Per draw type (``docs/huddles-peer-messages.md``):

    ``NEW``
        First point of a stroke: a **degenerate** segment (``a=b=c=d``). Later
        points: appends a segment. Cancels any pending fade.
    ``EDIT``
        Replaces the last segment of the sender's line. Repeating the last
        segment is a no-op for the picture, which is how keep-alives work.
    ``END``
        Finishes the stroke and starts a 700 ms timer before the 3 s fade.
    ``CHANGE_HUE``
        Recolours the sender's **whole current line**; pass ``hue``.
    ``POSITION``
        Moves the sender's pencil cursor and name tag to ``x``/``y``. It does
        not touch the line or either fade timer.

    Coordinates are normalized 0..1 over the shared video's bounding rect
    (``ProgressiveCurveTransformer``: ``[(x-left)/width, (y-top)/height]``).
    """
    if not item_id:
        raise ValueError("item_id is required and must be the sharer's attendeeId")
    msg = _message(from_peer_id)
    draw = msg.draw
    draw.type = draw_type
    draw.itemId = item_id
    if segment is not None:
        draw.segment.CopyFrom(_coerce_segment(segment))
    if hue is not None:
        draw.hue = validate_hue(hue)
    if x is not None:
        draw.x = float(x)
    if y is not None:
        draw.y = float(y)
    return _serialize(msg, what="draw")


# --------------------------------------------------------------------------
# capabilities (5) / contentShareDetails (15)
# --------------------------------------------------------------------------


class ContentShareSource(StrEnum):
    """The ``ContentShareSource`` values ``contentShareDetails.sourceType`` takes.

    Enumerated in ``docs/huddles-peer-messages.md``. Receivers store this as the
    peer's ``contentShareSource``; **drawing is only allowed for**
    :data:`DRAWABLE_SOURCE_TYPES` -- ``desktop:screen`` always, and
    ``desktop:window`` on desktop builds >= 4.30.79.
    """

    DESKTOP_SCREEN = "desktop:screen"
    DESKTOP_WINDOW = "desktop:window"
    DESKTOP_UNKNOWN = "desktop:unknown"
    BROWSER_CONTENT = "browser:content"
    MOBILE_SCREEN = "mobile:screen"
    UNKNOWN = "unknown"


#: Share sources Slack clients will render drawings over.
DRAWABLE_SOURCE_TYPES = frozenset(
    {ContentShareSource.DESKTOP_SCREEN, ContentShareSource.DESKTOP_WINDOW}
)


def build_capabilities(*, from_peer_id: str, is_draw_enabled: bool) -> bytes:
    """Build ``PeerMessage.capabilities`` -- the "drawing is on/off" toggle.

    The real client sends this when a screenshare starts or stops and on
    ``propagateDrawEnabled``. Receivers store ``isDrawEnabled`` on that peer's
    state (it also shows up as ``is_draw_enabled`` in the participant-state
    JSON schema).

    Caveat: the doc says receivers ignore the message "if the field is unset",
    and in proto3 a ``false`` bool is not written to the wire at all -- so
    ``is_draw_enabled=False`` serializes to an empty ``Capabilities``
    submessage. A protobuf-es receiver reads the missing field back as
    ``false``, which is the intended meaning, but there is no way to
    distinguish "explicitly off" from "unset" on the wire.
    """
    msg = _message(from_peer_id)
    msg.capabilities.isDrawEnabled = bool(is_draw_enabled)
    return _serialize(msg, what="capabilities")


def build_content_share_details(
    *,
    from_peer_id: str,
    source_type: ContentShareSource | str,
    item_id: str,
) -> bytes:
    """Build ``PeerMessage.contentShareDetails`` -- "here is what I am sharing".

    Sent together with ``capabilities`` when a share starts. ``item_id`` is the
    **sender's own attendeeId** (the sharer). Receivers set the peer's
    ``contentShareSource`` and ``isDrawEnabled`` from it; drawing is permitted
    only for ``desktop:screen`` and ``desktop:window`` (window needs desktop
    >= 4.30.79). Both ``itemId`` and ``sourceType`` are required by receivers.

    ``source_type`` is validated against :class:`ContentShareSource`; anything
    else would leave receivers with an unknown source and no drawing.
    """
    source = ContentShareSource(source_type)
    if not item_id:
        raise ValueError("item_id is required (the sharer's own attendeeId)")
    msg = _message(from_peer_id)
    msg.contentShareDetails.sourceType = str(source)
    msg.contentShareDetails.itemId = item_id
    return _serialize(msg, what="contentShareDetails")


# --------------------------------------------------------------------------
# reacji (1)
# --------------------------------------------------------------------------


def build_reacji(
    *,
    from_peer_id: str,
    emoji: str,
    reacji_type: ReacjiType.ValueType = ReacjiType.Standard,
    emoji_url: str | None = None,
) -> bytes:
    """Build ``PeerMessage.reacji`` -- a reaction on the sender's own tile.

    ``emoji`` is the bare name (no colons). Receivers show the reaction on the
    *sender's* tile -- there is no way to target someone else -- and log a
    ``SEND_REACTION`` activity event against the sender's userId. ``emojiUrl``
    is resolved through the client's own emoji-URL lookup, so it is optional.

    Checks on the receive side: the sender must be a known peer and ``emoji``
    must be set. Source-read only: no reacji was sent live.
    """
    if not emoji:
        raise ValueError("emoji is required (receivers drop a reacji without one)")
    msg = _message(from_peer_id)
    msg.reacji.emoji = emoji.strip(":")
    msg.reacji.type = reacji_type
    if emoji_url:
        msg.reacji.emojiUrl = emoji_url
    return _serialize(msg, what="reacji")


# --------------------------------------------------------------------------
# background (6)
# --------------------------------------------------------------------------


def build_background(*, from_peer_id: str, url: str) -> bytes:
    """Build ``PeerMessage.background`` -- a video background URL.

    **There is no receive branch for this message in the observed build (Slack
    4.52.x): it is sent and then ignored.** Nothing will happen on any
    receiver, so do not spend an hour debugging a no-op. The ``url`` format is
    an open question in the doc for exactly that reason -- no receiver to read
    it.
    """
    msg = _message(from_peer_id)
    msg.background.url = url
    return _serialize(msg, what="background")


# --------------------------------------------------------------------------
# muteRequest (10)
# --------------------------------------------------------------------------


def build_mute_request(*, from_peer_id: str, peer_id: str) -> bytes:
    """Build ``PeerMessage.muteRequest`` -- turn off someone else's microphone.

    **This acts on another person's client.** If ``peer_id`` is the receiver's
    own attendeeId, that client turns its microphone off (source
    ``PeerMuteRequest``) and shows a "peer muted you" alert; everyone else
    ignores the message.

    Safety (``docs/huddles-peer-messages.md#safety``): mute requests act on
    other people's clients -- **do not send this into a huddle whose
    participants have not agreed to it.** There is no technical gate here and
    none is faked: the caller is responsible for consent.

    Receive-side checks: the sender must be a known peer, and incoming mute
    requests are rate-limited per sender -- at most one per 10 s and at most 50
    in total. Source-read only: never sent live.
    """
    if not peer_id:
        raise ValueError("peer_id is required (the target's attendeeId)")
    msg = _message(from_peer_id)
    msg.muteRequest.peerId = peer_id
    return _serialize(msg, what="muteRequest")


# --------------------------------------------------------------------------
# kickNotification (18)
# --------------------------------------------------------------------------


def build_kick_notification(*, from_peer_id: str, kicked_user_id: str) -> bytes:
    """Build ``PeerMessage.kickNotification`` -- "X removed Y" plus kick handling.

    ``kicked_user_id`` is a Slack **user** id (``U…``), not an attendeeId.
    Everyone except the sender dispatches a "X removed Y" notice and runs the
    kick handling for that user.

    Safety (``docs/huddles-peer-messages.md#safety``): like ``muteRequest``,
    this acts on other people's clients -- **do not send it into a huddle whose
    participants have not agreed to it.** No safety gate is implemented here,
    deliberately; the caller carries the responsibility.

    Receive-side check: the sender must be a known peer. Source-read only:
    never sent live.
    """
    if not kicked_user_id:
        raise ValueError("kicked_user_id is required (a Slack user id, 'U…')")
    msg = _message(from_peer_id)
    msg.kickNotification.kickedUserId = kicked_user_id
    return _serialize(msg, what="kickNotification")


# --------------------------------------------------------------------------
# jukeboxRequest (14)
# --------------------------------------------------------------------------


def build_jukebox_request(
    *, from_peer_id: str, song_key: str = "", stop: bool = False
) -> bytes:
    """Build ``PeerMessage.jukeboxRequest`` -- play or stop a jukebox song.

    Pass either ``song_key`` or ``stop=True``. Only the current **DJ** acts on
    it, and only when the jukebox feature is enabled: it looks ``song_key`` up
    in the song table and plays or stops it. Everyone else ignores it.
    Receive-side check: sender must be a known peer. Source-read only.
    """
    if not song_key and not stop:
        raise ValueError("pass song_key, or stop=True")
    msg = _message(from_peer_id)
    if song_key:
        msg.jukeboxRequest.songKey = song_key
    msg.jukeboxRequest.stop = bool(stop)
    return _serialize(msg, what="jukeboxRequest")


# --------------------------------------------------------------------------
# recordingStartNotification (16)
# --------------------------------------------------------------------------


def build_recording_start_notification(
    *,
    from_peer_id: str,
    notification_type: RecordingType.ValueType = RecordingType.SUMMARY,
    recording_type: str | None = None,
) -> bytes:
    """Build ``PeerMessage.recordingStartNotification`` -- "recording started".

    **There is no receive branch for this message in the observed build (Slack
    4.52.x): it is sent and then ignored.** No receiver shows a banner, no
    peer state changes -- do not debug a no-op. The only documented ``type`` is
    ``SUMMARY``; ``recordingType`` is a free string whose meaning is unknown
    for the same reason (nothing reads it).
    """
    msg = _message(from_peer_id)
    msg.recordingStartNotification.type = notification_type
    if recording_type:
        msg.recordingStartNotification.recordingType = recording_type
    return _serialize(msg, what="recordingStartNotification")


# --------------------------------------------------------------------------
# awareness (11, 12, 13) -- topic awareness-sync-channel
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AwarenessField:
    """One ``AwarenessPayload`` entry: a field name, value and optional stamp.

    ``timestamp_ms`` is only included by the real client when
    ``isHuddlesAwarenessTimestampEnabled`` is set, and it is ``optional`` in the
    schema, so ``None`` means "do not write the field at all". The valid
    ``field_name`` values were not enumerated by the source read.
    """

    name: str
    value: str
    timestamp_ms: int | None = None


AwarenessState: TypeAlias = (
    "Mapping[str, str] | Iterable[AwarenessField] | Iterable[tuple[str, str]]"
)


def _awareness_fields(state: AwarenessState) -> list[AwarenessField]:
    if isinstance(state, Mapping):
        return [AwarenessField(str(k), str(v)) for k, v in state.items()]
    out: list[AwarenessField] = []
    for entry in state:
        if isinstance(entry, AwarenessField):
            out.append(entry)
        else:
            name, value, *rest = entry
            out.append(AwarenessField(str(name), str(value), rest[0] if rest else None))
    return out


def build_awareness_broadcast(
    *,
    from_peer_id: str,
    channel: str,
    user: str,
    state: AwarenessState = (),
) -> bytes:
    """Build ``PeerMessage.awarenessBroadcast`` -- per-channel presence state.

    Sent on topic ``awareness-sync-channel`` when the broadcaster has changes.
    The real client batches changes every 250 ms, flushes at most once per
    500 ms with 0-15 s of per-topic jitter, and does a full rebroadcast every
    30 s. Listeners dispatch it into the team store; the broadcaster itself
    ignores incoming messages. Source-read only.
    """
    msg = _message(from_peer_id)
    bcast = msg.awarenessBroadcast
    bcast.channel = channel
    bcast.user = user
    for item in _awareness_fields(state):
        payload = bcast.awarenessState.add()
        payload.field_name = item.name
        payload.field_value = item.value
        if item.timestamp_ms is not None:
            payload.field_timestamp_ms = int(item.timestamp_ms)
    return _serialize(msg, what="awarenessBroadcast")


def build_awareness_query(*, from_peer_id: str, channel: str, user: str) -> bytes:
    """Build ``PeerMessage.awarenessQuery`` -- "tell me the current state".

    Sent on ``awareness-sync-channel`` when a broadcaster connects, and at most
    once every 15 s after that. Listeners dispatch it into the team store.
    """
    msg = _message(from_peer_id)
    msg.awarenessQuery.channel = channel
    msg.awarenessQuery.user = user
    return _serialize(msg, what="awarenessQuery")


def build_awareness_close(*, from_peer_id: str, channel: str, user: str) -> bytes:
    """Build ``PeerMessage.awarenessClose`` -- "I am going away".

    Sent on ``awareness-sync-channel`` at broadcaster shutdown. Receiving
    listeners drop the sender's entries (they also expire on their own through a
    TTL sweep every 5 s).
    """
    msg = _message(from_peer_id)
    msg.awarenessClose.channel = channel
    msg.awarenessClose.user = user
    return _serialize(msg, what="awarenessClose")


# --------------------------------------------------------------------------
# transcriptCatchUp (17) -- topic transcript-catchup
# --------------------------------------------------------------------------

#: A single ``RESPONSE`` must stay under this; above it, chunk.
TRANSCRIPT_RESPONSE_MAX_BYTES = 2048
#: Each ``RESPONSE_CHUNK`` must stay under this (tighter than a plain RESPONSE).
TRANSCRIPT_CHUNK_MAX_BYTES = 2032


@dataclass(frozen=True, slots=True)
class TranscriptLine:
    """One transcript line, mirroring ``PeerMessage.TranscriptLine``."""

    user_id: str
    content: str
    start_time_ms: int = 0
    end_time_ms: int = 0


LineLike: TypeAlias = "TranscriptLine | pb.PeerMessage.TranscriptLine"


def _as_line(line: LineLike) -> TranscriptLine:
    if isinstance(line, TranscriptLine):
        return line
    return TranscriptLine(
        user_id=line.user_id,
        content=line.content,
        start_time_ms=line.start_time_ms,
        end_time_ms=line.end_time_ms,
    )


def make_request_id(attendee_id: str, *, probe: bool = False, now_ms: int | None = None) -> str:
    """Build the ``requestId`` a ``SOLICIT`` carries.

    The doc: the joiner sends ``SOLICIT {requestId: "<self>-<ms>"}``, and a
    probe uses ``"<self>-probe-<ms>"``.
    """
    ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return f"{attendee_id}-probe-{ms}" if probe else f"{attendee_id}-{ms}"


def make_request_from_id(chosen_attendee_id: str, request_id: str) -> str:
    """Build the ``REQUEST_FROM`` id: ``"<chosenAttendeeId>:<requestId>"``.

    The chosen peer only answers if the prefix before ``:`` is its own
    attendeeId, which is how one responder is picked out of a broadcast.
    """
    return f"{chosen_attendee_id}:{request_id}"


def build_transcript_catch_up(
    *,
    from_peer_id: str,
    catch_up_type: TranscriptType.ValueType,
    request_id: str = "",
    lines: Sequence[LineLike] = (),
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    earliest_transcript_timestamp_ms: int | None = None,
    limit: int = MAX_DATA_MESSAGE_BYTES,
) -> bytes:
    """Build ``PeerMessage.transcriptCatchUp`` -- one step of the catch-up flow.

    The flow (topic ``transcript-catchup``, for late joiners fetching earlier
    transcript):

    1. The joiner sends ``SOLICIT {requestId: "<self>-<ms>"}``
       (:func:`make_request_id`).
    2. Every peer that has lines replies
       ``OFFER {requestId, earliestTranscriptTimestampMs}``.
    3. After 2 s the joiner sorts the offers (earliest first) and sends
       ``REQUEST_FROM {requestId: "<chosenAttendeeId>:<requestId>"}``
       (:func:`make_request_from_id`).
    4. The chosen peer replies with lines from the **last 20 minutes**: one
       ``RESPONSE`` if the wire size is under 2048 bytes, else several
       ``RESPONSE_CHUNK`` messages each under 2032 bytes
       (:func:`build_transcript_response`).
    5. The joiner waits 5 s per chunk and tries up to 3 candidates.

    ``limit`` lets the chunker apply the tighter 2032-byte budget.
    Source-read only: the flow was never exercised live.
    """
    msg = _message(from_peer_id)
    catch_up = msg.transcriptCatchUp
    catch_up.type = catch_up_type
    if request_id:
        catch_up.request_id = request_id
    for line in lines:
        value = _as_line(line)
        out = catch_up.lines.add()
        out.user_id = value.user_id
        out.content = value.content
        out.start_time_ms = int(value.start_time_ms)
        out.end_time_ms = int(value.end_time_ms)
    if chunk_index is not None:
        catch_up.chunk_index = int(chunk_index)
    if total_chunks is not None:
        catch_up.total_chunks = int(total_chunks)
    if earliest_transcript_timestamp_ms is not None:
        catch_up.earliest_transcript_timestamp_ms = int(earliest_transcript_timestamp_ms)
    return _serialize(msg, limit=limit, what="transcriptCatchUp")


def build_transcript_solicit(
    *, from_peer_id: str, request_id: str | None = None, probe: bool = False
) -> bytes:
    """Step 1: ``SOLICIT`` -- "does anyone have earlier transcript?"."""
    return build_transcript_catch_up(
        from_peer_id=from_peer_id,
        catch_up_type=TranscriptType.SOLICIT,
        request_id=request_id or make_request_id(from_peer_id, probe=probe),
    )


def build_transcript_offer(
    *,
    from_peer_id: str,
    request_id: str,
    earliest_transcript_timestamp_ms: int,
) -> bytes:
    """Step 2: ``OFFER`` -- "I have lines back to this timestamp"."""
    return build_transcript_catch_up(
        from_peer_id=from_peer_id,
        catch_up_type=TranscriptType.OFFER,
        request_id=request_id,
        earliest_transcript_timestamp_ms=earliest_transcript_timestamp_ms,
    )


def build_transcript_request_from(
    *, from_peer_id: str, chosen_attendee_id: str, request_id: str
) -> bytes:
    """Step 3: ``REQUEST_FROM`` -- "you specifically, please send them".

    The id sent is ``"<chosenAttendeeId>:<requestId>"``; every other peer sees
    the broadcast and ignores it because the prefix is not its own attendeeId.
    """
    return build_transcript_catch_up(
        from_peer_id=from_peer_id,
        catch_up_type=TranscriptType.REQUEST_FROM,
        request_id=make_request_from_id(chosen_attendee_id, request_id),
    )


def _line_overhead(from_peer_id: str, request_id: str, line: TranscriptLine) -> int:
    """Bytes a chunk costs with ``line`` present but its content empty."""
    probe = TranscriptLine(line.user_id, "", line.start_time_ms, line.end_time_ms)
    return len(
        build_transcript_catch_up(
            from_peer_id=from_peer_id,
            catch_up_type=TranscriptType.RESPONSE_CHUNK,
            request_id=request_id,
            lines=[probe],
            # Worst-case 5-byte varints, so a real index can only be smaller.
            chunk_index=0xFFFFFFFF,
            total_chunks=0xFFFFFFFF,
            limit=TRANSCRIPT_CHUNK_MAX_BYTES * 4,
        )
    )


def _split_text(text: str, budget: int) -> list[str]:
    """Split ``text`` into pieces whose UTF-8 encodings each fit ``budget``.

    Splits on a character boundary (never mid code point) and prefers the last
    space in a piece so chunk boundaries fall between words where possible.
    """
    if budget <= 0:
        raise ValueError("no room left for transcript content; request_id too long?")
    pieces: list[str] = []
    rest = text
    while len(rest.encode("utf-8")) > budget:
        # Walk back from an upper bound until the encoding fits.
        cut = min(len(rest), budget)
        while cut > 1 and len(rest[:cut].encode("utf-8")) > budget:
            cut -= 1
        space = rest.rfind(" ", 1, cut)
        if space > cut // 2:
            cut = space + 1
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces or [""]


def split_transcript_lines(
    *,
    from_peer_id: str,
    request_id: str,
    lines: Sequence[LineLike],
    limit: int = TRANSCRIPT_CHUNK_MAX_BYTES,
) -> list[list[TranscriptLine]]:
    """Group ``lines`` so each ``RESPONSE_CHUNK`` stays under ``limit`` bytes.

    Doc step 4: the chosen peer sends "several ``RESPONSE_CHUNK {lines,
    chunkIndex, totalChunks, requestId}`` messages, each under 2032 bytes (long
    lines are split)". So a single line longer than one chunk is split by
    content, keeping its ``user_id`` and timestamps on every piece -- the
    receiver reassembles by concatenating lines in chunk order.

    Sizes are measured against real serialized messages with worst-case
    (5-byte varint) ``chunkIndex``/``totalChunks``, so the messages actually
    sent can only be smaller.
    """
    expanded: list[TranscriptLine] = []
    for raw in lines:
        line = _as_line(raw)
        budget = limit - 1 - _line_overhead(from_peer_id, request_id, line)
        # -1 keeps "under limit" strict rather than "at most limit".
        content_budget = budget - 8  # length prefix + field tag slack
        if len(line.content.encode("utf-8")) <= content_budget:
            expanded.append(line)
            continue
        for piece in _split_text(line.content, content_budget):
            expanded.append(
                TranscriptLine(
                    line.user_id, piece, line.start_time_ms, line.end_time_ms
                )
            )

    groups: list[list[TranscriptLine]] = []
    current: list[TranscriptLine] = []

    def fits(candidate: list[TranscriptLine]) -> bool:
        size = len(
            build_transcript_catch_up(
                from_peer_id=from_peer_id,
                catch_up_type=TranscriptType.RESPONSE_CHUNK,
                request_id=request_id,
                lines=candidate,
                chunk_index=0xFFFFFFFF,
                total_chunks=0xFFFFFFFF,
                limit=limit * 8,
            )
        )
        return size < limit

    for line in expanded:
        if current and not fits([*current, line]):
            groups.append(current)
            current = []
        current.append(line)
        if not fits(current):  # pragma: no cover - guarded by the split above
            raise PayloadTooLarge(limit, limit, "transcript line after splitting")
    if current:
        groups.append(current)
    return groups


def build_transcript_response(
    *,
    from_peer_id: str,
    request_id: str,
    lines: Sequence[LineLike],
) -> list[bytes]:
    """Step 4: the transcript reply, as one ``RESPONSE`` or N ``RESPONSE_CHUNK``.

    Doc: "It sends one ``RESPONSE {lines, requestId}`` if the wire size is under
    2048 bytes, otherwise several ``RESPONSE_CHUNK {lines, chunkIndex,
    totalChunks, requestId}`` messages, each under 2032 bytes (long lines are
    split)." ``chunkIndex`` is emitted 0-based; note that proto3 does not write
    a zero varint, so chunk 0 carries no ``chunk_index`` field and the receiver
    reads the default 0.

    The lines passed in should already be limited to the last 20 minutes, which
    is what the real client replies with.
    """
    single = build_transcript_catch_up(
        from_peer_id=from_peer_id,
        catch_up_type=TranscriptType.RESPONSE,
        request_id=request_id,
        lines=lines,
        limit=MAX_DATA_MESSAGE_BYTES * 8,  # measure first, decide after
    )
    if len(single) < TRANSCRIPT_RESPONSE_MAX_BYTES:
        return [single]

    groups = split_transcript_lines(
        from_peer_id=from_peer_id, request_id=request_id, lines=lines
    )
    total = len(groups)
    logger.debug(
        "transcript response %s: %d lines -> %d chunks", request_id, len(lines), total
    )
    return [
        build_transcript_catch_up(
            from_peer_id=from_peer_id,
            catch_up_type=TranscriptType.RESPONSE_CHUNK,
            request_id=request_id,
            lines=group,
            chunk_index=index,
            total_chunks=total,
            limit=TRANSCRIPT_CHUNK_MAX_BYTES,
        )
        for index, group in enumerate(groups)
    ]


# --------------------------------------------------------------------------
# parse / describe
# --------------------------------------------------------------------------

#: Payload fields of PeerMessage, in field-number order.
PAYLOAD_FIELDS: tuple[str, ...] = (
    "reacji",
    "draw",
    "capabilities",
    "background",
    "muteRequest",
    "awarenessBroadcast",
    "awarenessQuery",
    "awarenessClose",
    "jukeboxRequest",
    "contentShareDetails",
    "recordingStartNotification",
    "transcriptCatchUp",
    "kickNotification",
)


def parse_peer_message(data: bytes) -> PeerMessage:
    """Parse a received data-message payload.

    Raises ``google.protobuf.message.DecodeError`` on garbage -- callers that
    receive from the wire should expect that, since any attendee can send
    anything on any topic.
    """
    msg = PeerMessage()
    msg.ParseFromString(bytes(data))
    return msg


def _fmt_point(point: ProtoPoint) -> str:
    return f"({point.x:.3f},{point.y:.3f})"


def _describe_draw(draw: pb.PeerMessage.Draw) -> str:
    name = DrawType.Name(draw.type)
    parts = [f"draw {name} item={draw.itemId or '-'}"]
    if draw.type == DrawType.CHANGE_HUE or draw.hue:
        parts.append(f"hue={draw.hue}")
    if draw.type == DrawType.POSITION:
        parts.append(f"at=({draw.x:.3f},{draw.y:.3f})")
    if draw.HasField("segment"):
        seg = draw.segment
        points = [seg.a, seg.b, seg.c, seg.d]
        if all(
            p.x == seg.a.x and p.y == seg.a.y for p in points
        ):  # a=b=c=d: the first point of a stroke
            parts.append(f"degenerate{_fmt_point(seg.a)}")
        else:
            parts.append(f"{_fmt_point(seg.a)}->{_fmt_point(seg.d)}")
    return " ".join(parts)


def describe_peer_message(msg: PeerMessage | bytes) -> str:
    """One-line, log-friendly summary of a ``PeerMessage``.

    Accepts the parsed message or the raw bytes. Keeps coordinates short and
    does not dump transcript text, so it is safe to log at debug level in a
    tight loop.
    """
    if isinstance(msg, (bytes, bytearray, memoryview)):
        msg = parse_peer_message(bytes(msg))
    sender = msg.fromPeerIdString or (str(msg.fromPeerId) if msg.fromPeerId else "?")
    parts: list[str] = []
    for name in PAYLOAD_FIELDS:
        if not msg.HasField(name):
            continue
        value = getattr(msg, name)
        match name:
            case "draw":
                parts.append(_describe_draw(value))
            case "reacji":
                parts.append(
                    f"reacji :{value.emoji}: {ReacjiType.Name(value.type)}"
                )
            case "capabilities":
                parts.append(f"capabilities isDrawEnabled={value.isDrawEnabled}")
            case "background":
                parts.append(f"background url={value.url!r} (no receiver)")
            case "muteRequest":
                parts.append(f"muteRequest target={value.peerId}")
            case "jukeboxRequest":
                parts.append(
                    "jukeboxRequest stop"
                    if value.stop
                    else f"jukeboxRequest song={value.songKey}"
                )
            case "contentShareDetails":
                parts.append(
                    f"contentShareDetails {value.sourceType} item={value.itemId}"
                )
            case "recordingStartNotification":
                parts.append(
                    f"recordingStartNotification {RecordingType.Name(value.type)}"
                    f"{'/' + value.recordingType if value.recordingType else ''}"
                    " (no receiver)"
                )
            case "kickNotification":
                parts.append(f"kickNotification user={value.kickedUserId}")
            case "transcriptCatchUp":
                desc = f"transcriptCatchUp {TranscriptType.Name(value.type)}"
                if value.request_id:
                    desc += f" req={value.request_id}"
                if value.lines:
                    desc += f" lines={len(value.lines)}"
                if value.total_chunks:
                    desc += f" chunk={value.chunk_index + 1}/{value.total_chunks}"
                if value.earliest_transcript_timestamp_ms:
                    desc += f" earliest={value.earliest_transcript_timestamp_ms}"
                parts.append(desc)
            case "awarenessBroadcast":
                parts.append(
                    f"awarenessBroadcast {value.channel}/{value.user} "
                    f"fields={len(value.awarenessState)}"
                )
            case "awarenessQuery":
                parts.append(f"awarenessQuery {value.channel}/{value.user}")
            case "awarenessClose":
                parts.append(f"awarenessClose {value.channel}/{value.user}")
    body = ", ".join(parts) if parts else "empty"
    return f"PeerMessage(from={sender}, {body})"
