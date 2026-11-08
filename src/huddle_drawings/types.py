"""Shared data types.

These are the contract between the Slack layer (``slack.py``), the Chime
signaling layer (``signaling.py``), the media layer (``audio.py``) and the
drawing layer (``draw.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MediaPlacement:
    """Chime ``MediaPlacement`` -- the endpoint set for one meeting.

    AWS returns PascalCase keys; we keep snake_case here.
    """

    audio_host_url: str
    signaling_url: str
    turn_control_url: str | None = None
    screen_data_url: str | None = None
    screen_sharing_url: str | None = None
    screen_viewing_url: str | None = None
    event_ingestion_url: str | None = None

    @classmethod
    def from_aws(cls, data: dict[str, Any]) -> MediaPlacement:
        return cls(
            audio_host_url=data["AudioHostUrl"],
            signaling_url=data["SignalingUrl"],
            turn_control_url=data.get("TurnControlUrl"),
            screen_data_url=data.get("ScreenDataUrl"),
            screen_sharing_url=data.get("ScreenSharingUrl"),
            screen_viewing_url=data.get("ScreenViewingUrl"),
            event_ingestion_url=data.get("EventIngestionUrl"),
        )


@dataclass(frozen=True, slots=True)
class ChimeCredentials:
    """The ``call.free_willy`` blob from ``rooms.join``, flattened.

    ``join_token`` is a secret and should be treated like a session token.
    The same user gets the same ``attendee_id`` and ``join_token`` on every
    ``rooms.join`` while the room lives -- so one account is always one
    attendee, which is one drawing line in one hue.
    """

    meeting_id: str
    media_region: str
    external_meeting_id: str | None
    media_placement: MediaPlacement
    attendee_id: str
    external_user_id: str
    join_token: str
    #: Raw Chime objects, as returned, for anything not modelled above.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_free_willy(cls, free_willy: dict[str, Any]) -> ChimeCredentials:
        meeting = free_willy["meeting"]
        attendee = free_willy["attendee"]
        return cls(
            meeting_id=meeting["MeetingId"],
            media_region=meeting.get("MediaRegion", ""),
            external_meeting_id=meeting.get("ExternalMeetingId"),
            media_placement=MediaPlacement.from_aws(meeting["MediaPlacement"]),
            attendee_id=attendee["AttendeeId"],
            external_user_id=attendee.get("ExternalUserId", ""),
            join_token=attendee["JoinToken"],
            raw=free_willy,
        )

    def __repr__(self) -> str:  # keep the join token out of logs
        return (
            f"ChimeCredentials(meeting_id={self.meeting_id!r}, "
            f"media_region={self.media_region!r}, "
            f"attendee_id={self.attendee_id!r}, join_token=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class HuddleJoin:
    """Everything useful from a ``rooms.join`` response."""

    call_id: str
    chime: ChimeCredentials
    huddle_id: str | None = None
    created_by: str | None = None
    participants: tuple[str, ...] = ()
    thread_channel_id: str | None = None
    thread_root_ts: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class StreamSource:
    """One media source as listed in a Chime ``INDEX`` frame.

    A screen share's ``attendee_id`` is ``<sharerAttendeeId>#content``; the
    ``itemId`` to draw with is that with ``#content`` stripped, which is what
    :attr:`sharer_attendee_id` gives.
    """

    stream_id: int
    attendee_id: str
    external_user_id: str | None = None
    media_type: str | None = None  # "AUDIO" | "VIDEO"
    width: int | None = None
    height: int | None = None
    group_id: int | None = None
    max_bitrate_kbps: int | None = None

    @property
    def is_content_share(self) -> bool:
        return self.attendee_id.endswith("#content")

    @property
    def sharer_attendee_id(self) -> str:
        """The attendee behind this source, with any ``#content`` suffix removed.

        This is the value to use as a draw message's ``itemId``.
        """
        return self.attendee_id.removesuffix("#content")


@dataclass(frozen=True, slots=True)
class AttendeePresence:
    """An entry from a Chime ``AUDIO_STREAM_ID_INFO`` frame.

    Slack clients only learn about attendees Chime reports here, and that only
    happens once the attendee's audio connection is up. Drawings from an
    attendee nobody has seen in one of these frames are dropped.
    """

    audio_stream_id: int
    attendee_id: str
    external_user_id: str | None = None
    muted: bool = False
    dropped: bool = False


@dataclass(frozen=True, slots=True)
class DataMessage:
    """A received Chime data message payload."""

    topic: str
    data: bytes
    sender_attendee_id: str
    sender_external_user_id: str | None = None
    lifetime_ms: int = 0
    ingest_time_ns: int | None = None


class HuddleError(RuntimeError):
    """Base class for this package's errors."""


class SlackApiError(HuddleError):
    """A Slack Web API call returned ``ok: false``."""

    def __init__(self, method: str, error: str, response: dict[str, Any]) -> None:
        super().__init__(f"{method} failed: {error}")
        self.method = method
        self.error = error
        self.response = response


class SignalingError(HuddleError):
    """The Chime signaling channel failed or closed unexpectedly.

    Close codes seen in the wild: ``4410`` the meeting has ended;
    ``4403`` "attendee unavailable" (often a previous connection for the same
    account that did not leave cleanly); ``1006`` dropped -- bad SRTP profile,
    or data messages far over the rate limit.
    """

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


CLOSE_CODE_MEANINGS = {
    1006: (
        "dropped by Chime. In rough order of likelihood: the WebRTC media leg "
        "never connected (Chime media only traverses Chime's own TURN relay, "
        "from the JOIN_ACK credentials, with relay-only ICE); no audio is "
        "actually flowing (Chime drops an attendee a couple of seconds in); "
        "the DTLS-SRTP profile is not SRTP_AES128_CM_HMAC_SHA1_80; or data "
        "messages went far over the rate limit"
    ),
    4403: "attendee unavailable (a previous connection for this account may not have left cleanly)",
    4410: "the meeting has ended",
}
