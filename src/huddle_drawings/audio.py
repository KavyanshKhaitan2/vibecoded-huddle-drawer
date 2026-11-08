"""The required WebRTC media leg (aiortc).

Signaling alone lets an attendee send and receive Chime data messages, but
Slack clients drop drawings from attendees they do not know, and they only
learn about an attendee once Chime reports it in ``AUDIO_STREAM_ID_INFO`` --
which only happens once real audio is flowing
(``docs/huddles-and-chime.md`` step 3, ``docs/huddles-peer-messages.md``
"Sending"). So this module exists to make the bot *present*: an
:class:`RTCPeerConnection` that sends 20 ms Opus packets of near-silence and
can receive the huddle's mixed audio and a share's H.264 video.

Three protocol rules from the doc are implemented here and must not be
"simplified" away:

1. **The DTLS-SRTP profile must be ``SRTP_AES128_CM_HMAC_SHA1_80``.** aiortc
   (like werift) offers AES-GCM first; Chime's audio server cannot decrypt it,
   sends nothing back and drops the attendee ~2.5 s after the first packet
   (the signaling websocket then closes with 1006). See
   :func:`force_srtp_profile`.
2. **The offer needs a ``recvonly`` video m-section** even when video is not
   wanted: without one, ``SUBSCRIBE_ACK`` fails with "failed to initialize
   video session".
3. **Audio must be paced off the wall clock.** A naive ``sleep(0.02)`` on a
   busy event loop fires late, the 20 ms grid slips and the audio comes out
   crackly. See :class:`WallClockPacer`.
"""

from __future__ import annotations

import asyncio
import fractions
import inspect
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Final

import av
from aiortc import (
    RTCBundlePolicy,
    RTCConfiguration,
    RTCIceTransportPolicy,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpCodecCapability,
    RTCRtpReceiver,
    RTCRtpTransceiver,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from .types import HuddleError

logger = logging.getLogger(__name__)

#: Opus as Chime expects it in the offer: ``a=rtpmap:111 opus/48000/2`` with
#: ``a=fmtp:111 minptime=10;useinbandfec=1`` (doc step 3).
OPUS_PAYLOAD_TYPE: Final = 111


#: Cipher suites to offer on the DTLS handshake.
#:
#: aiortc offers **only** ``ECDHE-ECDSA-*``. We are the DTLS *client*, so the
#: negotiated suite has to match **Chime's** server certificate, and Chime's
#: media server presents an **RSA** certificate -- so with aiortc's stock list
#: there is no common suite and the server answers with a fatal
#: ``handshake_failure`` (alert 40). Verified live: adding the ECDHE-RSA
#: suites below moved the handshake past suite selection on the first try.
#: This list mirrors what Chrome offers.
CHIME_DTLS_CIPHERS: Final = (
    b"ECDHE-ECDSA-AES128-GCM-SHA256:"
    b"ECDHE-RSA-AES128-GCM-SHA256:"
    b"ECDHE-ECDSA-AES256-GCM-SHA384:"
    b"ECDHE-RSA-AES256-GCM-SHA384:"
    b"ECDHE-ECDSA-CHACHA20-POLY1305:"
    b"ECDHE-RSA-CHACHA20-POLY1305:"
    b"ECDHE-ECDSA-AES128-SHA:"
    b"ECDHE-RSA-AES128-SHA:"
    b"ECDHE-ECDSA-AES256-SHA:"
    b"ECDHE-RSA-AES256-SHA"
)

_ciphers_forced = False


def force_chime_dtls_ciphers() -> None:
    """Broaden aiortc's DTLS cipher list so Chime's RSA certificate works.

    aiortc hardcodes an ECDHE-ECDSA-only cipher list in
    ``RTCCertificate._create_ssl_context``. Since we are the DTLS client, the
    suite must suit *Chime's* certificate, which is RSA -- so the stock list
    yields no overlap and Chime replies with a fatal ``handshake_failure``
    (alert 40) about a second after ICE completes. Because Chime then closes
    the signaling websocket too, the visible symptom is a 1006 close that
    looks like an SRTP or rate-limit problem and is neither.

    This wraps the context factory and re-applies a Chrome-like cipher list
    (the later ``set_cipher_list`` wins). Idempotent, and process-wide, like
    the other aiortc adjustments here.

    :raises RuntimeError: if aiortc's internals have moved, since a silent
        failure here costs a mysterious 1006.
    """
    global _ciphers_forced
    if _ciphers_forced:
        return
    try:
        from aiortc.rtcdtlstransport import RTCCertificate
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"cannot import aiortc's RTCCertificate: {exc}") from exc

    original = getattr(RTCCertificate, "_create_ssl_context", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "aiortc's RTCCertificate has no _create_ssl_context; its DTLS "
            "internals have changed and Chime's RSA certificate will be "
            "rejected with a fatal handshake_failure"
        )

    def _create_ssl_context(self: Any, srtp_profiles: Any) -> Any:
        ctx = original(self, srtp_profiles)
        ctx.set_cipher_list(CHIME_DTLS_CIPHERS)
        return ctx

    _create_ssl_context._huddle_drawings_patched = True  # type: ignore[attr-defined]
    RTCCertificate._create_ssl_context = _create_ssl_context  # type: ignore[method-assign]
    _ciphers_forced = True
    logger.info(
        "broadened aiortc's DTLS cipher list to include ECDHE-RSA suites "
        "(Chime's media server uses an RSA certificate; aiortc offers only "
        "ECDHE-ECDSA and gets a fatal handshake_failure)"
    )


def keep_single_fingerprint(sdp: str, algorithm: str = "sha-256") -> str:
    """Keep only ``algorithm`` DTLS fingerprints in an SDP.

    aiortc advertises **three** fingerprints per m-section (sha-256, sha-384
    and sha-512). Browsers send exactly one, and Chime rejects the multi-
    fingerprint offer during certificate verification with a fatal
    ``unknown_ca`` (alert 48).

    Verified live: with the broadened cipher list the handshake reached
    certificate verification and failed with alert 48; trimming to the single
    sha-256 line made the media leg connect immediately. RFC 8122 does permit
    several fingerprints, so this is Chime being strict rather than aiortc
    being wrong -- but Chime is the one we have to talk to.
    """
    prefix = f"a=fingerprint:{algorithm}"
    lines = sdp.splitlines()
    kept = [
        line
        for line in lines
        if not line.startswith("a=fingerprint:") or line.startswith(prefix)
    ]
    if not any(line.startswith(prefix) for line in kept):
        logger.warning(
            "SDP has no %s fingerprint; leaving its fingerprints untouched", algorithm
        )
        return sdp
    dropped = len(lines) - len(kept)
    if not dropped:
        return sdp
    logger.info(
        "dropped %d non-%s DTLS fingerprint line(s) (Chime rejects a "
        "multi-fingerprint offer with a fatal unknown_ca)",
        dropped,
        algorithm,
    )
    sep = "\r\n" if "\r\n" in sdp else "\n"
    trailing = sep if sdp.endswith(("\r\n", "\n")) else ""
    return sep.join(kept) + trailing


def propagate_bundle_candidates(sdp: str) -> str:
    """Copy the primary bundled m-section's ICE candidates onto the others.

    Works around a real aiortc limitation. In
    ``RTCPeerConnection.setRemoteDescription`` (aiortc 1.16,
    ``rtcpeerconnection.py`` ~line 988) remote candidates are collected as::

        iceCandidates[iceTransport] = media

    Under ``max-bundle`` every m-section shares **one** ICE transport, so that
    assignment overwrites rather than accumulates, and only the *last*
    section's candidates survive. A correctly-behaving bundling answerer --
    Chime included -- puts its candidates only on the primary section and
    leaves the rest with ``c=IN IP4 0.0.0.0`` and no candidates. The surviving
    entry is therefore empty, no remote candidates are ever added, and ICE sits
    in ``checking`` until the peer gives up.

    Duplicating the candidate lines into every bundled section makes the
    overwrite harmless whichever section lands last. Candidates are additive
    and identical here by definition (that is what bundling means), so this
    changes nothing semantically.

    Returns the SDP unchanged if there is no BUNDLE group or nothing to copy.
    """
    lines = sdp.splitlines()
    # Split into a session part and the m-sections that follow it.
    first_m = next((i for i, l in enumerate(lines) if l.startswith("m=")), None)
    if first_m is None:
        return sdp
    header, sections = lines[:first_m], []
    current: list[str] = []
    for line in lines[first_m:]:
        if line.startswith("m="):
            if current:
                sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    if not any(l.startswith("a=group:BUNDLE") for l in header):
        return sdp

    donor = next(
        (s for s in sections if any(l.startswith("a=candidate:") for l in s)), None
    )
    if donor is None:
        return sdp
    candidates = [l for l in donor if l.startswith("a=candidate:")]

    changed = False
    for section in sections:
        if section is donor:
            continue
        if any(l.startswith("a=candidate:") for l in section):
            continue
        # Insert after the last attribute we must not get in front of; simply
        # appending to the section is valid SDP and is what browsers emit.
        insert_at = len(section)
        for idx, line in enumerate(section):
            if line.startswith("a=end-of-candidates"):
                insert_at = idx
                break
        section[insert_at:insert_at] = candidates
        changed = True

    if not changed:
        return sdp
    logger.info(
        "copied %d ICE candidate line(s) from the primary bundled m-section "
        "onto %d other section(s) (works around aiortc's max-bundle candidate "
        "handling)",
        len(candidates),
        sum(1 for s in sections if s is not donor),
    )
    out = header + [l for section in sections for l in section]
    trailing = "\r\n" if sdp.endswith("\r\n") else "\n" if sdp.endswith("\n") else ""
    sep = "\r\n" if "\r\n" in sdp else "\n"
    return sep.join(out) + trailing


class MediaConnectionError(HuddleError):
    """The WebRTC media leg failed to come up.

    Distinct from a :class:`~huddle_drawings.types.SignalingError`: this says
    *our* media never connected, rather than that Chime hung up on us.
    """


def _is_turn_server(server: RTCIceServer) -> bool:
    """Whether an ICE server entry is a TURN (rather than STUN) server."""
    urls = [server.urls] if isinstance(server.urls, str) else list(server.urls)
    return any(str(u).lower().startswith(("turn:", "turns:")) for u in urls)


def ice_servers_from_turn_credentials(turn: Any) -> list[RTCIceServer]:
    """Build ICE servers from a Chime ``JOIN_ACK``'s ``SdkTurnCredentials``.

    Chime's media server is reachable **only** through these; see
    :class:`AudioLeg` for what happens otherwise. ``turn`` is the
    ``SdkTurnCredentials`` message (or anything with ``uris``, ``username``
    and ``password``); ``None`` or a credentialless blob gives ``[]``.

    aiortc hands a single TURN server to aioice, so the URIs are passed as one
    entry and aiortc picks from them.
    """
    if turn is None:
        return []
    uris = [str(u) for u in getattr(turn, "uris", []) if u]
    username = getattr(turn, "username", "") or ""
    password = getattr(turn, "password", "") or ""
    if not uris:
        logger.warning("JOIN_ACK carried no TURN uris; Chime media will not connect")
        return []
    if not username or not password:
        logger.warning(
            "JOIN_ACK TURN credentials are missing a username or password; "
            "Chime media will probably not connect"
        )
    logger.info(
        "using %d Chime TURN uri(s) for the media leg: %s",
        len(uris),
        ", ".join(uris),
    )
    return [RTCIceServer(urls=uris, username=username, credential=password)]
OPUS_SAMPLE_RATE: Final = 48000
OPUS_CHANNELS: Final = 2
OPUS_FMTP: Final[dict[str, Any]] = {"minptime": 10, "useinbandfec": 1}

#: 20 ms packetization, the only ptime Chime's audio server is known to take.
AUDIO_PTIME: Final = 0.020
SAMPLES_PER_FRAME: Final = int(OPUS_SAMPLE_RATE * AUDIO_PTIME)  # 960

#: OpenSSL name of the one SRTP profile Chime can decrypt.
SRTP_PROFILE_NAME: Final = b"SRTP_AES128_CM_SHA1_80"

#: The Chime JS SDK rewrites the offer's ``o=-`` line to ``o=mozilla-chrome``
#: before sending ``SUBSCRIBE`` (doc step 3); we mimic it by default.
SDP_ORIGIN_USERNAME: Final = "mozilla-chrome"

TrackCallback = Callable[[MediaStreamTrack], Any]


# --------------------------------------------------------------------------- #
# The SRTP gotcha
# --------------------------------------------------------------------------- #


def force_srtp_profile() -> None:
    """Restrict aiortc to ``SRTP_AES128_CM_SHA1_80`` -- mandatory, see the doc.

    ``docs/huddles-and-chime.md`` step 3, "SRTP gotcha": the DTLS-SRTP profile
    must be ``SRTP_AES128_CM_HMAC_SHA1_80``. aiortc builds its offered profile
    list at import time as
    ``aiortc.rtcdtlstransport.SRTP_PROFILES = [AEAD_AES_256_GCM,
    AEAD_AES_128_GCM, AES128_CM_SHA1_80]`` and every
    :class:`~aiortc.RTCDtlsTransport` takes a *reference* to that very list
    (``self._srtp_profiles = SRTP_PROFILES``), passing it to
    ``SSL.Context.set_tlsext_use_srtp()``. So AES-GCM would win the
    negotiation, Chime's audio server would be unable to decrypt our RTP, send
    nothing back, and drop the attendee ~2.5 s after the first packet (the
    websocket closing with 1006 -- a failure that looks like anything but a
    cipher problem).

    The list is therefore mutated **in place**, so transports and modules that
    already hold a reference see the restriction too. Idempotent: calling it
    again is a no-op.

    :raises RuntimeError: if aiortc's internals no longer have the shape this
        relies on (missing names, a non-list, a renamed profile, a
        ``RTCDtlsTransport.__init__`` that no longer reads the module-level
        list, or a libsrtp build without the profile). Raising is deliberate:
        silently failing here costs the user a mysterious 1006 drop.
    """
    try:
        from aiortc import rtcdtlstransport
    except ImportError as exc:  # pragma: no cover - aiortc is a hard dep
        raise RuntimeError(f"cannot import aiortc.rtcdtlstransport: {exc}") from exc

    profiles = getattr(rtcdtlstransport, "SRTP_PROFILES", None)
    wanted = getattr(rtcdtlstransport, "SRTP_AES128_CM_SHA1_80", None)
    if profiles is None or wanted is None:
        raise RuntimeError(
            "aiortc.rtcdtlstransport no longer exposes SRTP_PROFILES / "
            "SRTP_AES128_CM_SHA1_80; the mandatory Chime SRTP profile cannot "
            "be forced (expect Chime to drop the attendee with close 1006)"
        )
    if not isinstance(profiles, list):
        raise RuntimeError(  # noqa: TRY004 - one error type for "aiortc changed"
            "aiortc.rtcdtlstransport.SRTP_PROFILES is no longer a list but "
            f"{type(profiles).__name__}; cannot restrict it in place"
        )
    if getattr(wanted, "openssl_profile", None) != SRTP_PROFILE_NAME:
        raise RuntimeError(
            "aiortc's SRTP_AES128_CM_SHA1_80 does not carry openssl_profile "
            f"{SRTP_PROFILE_NAME!r} (got "
            f"{getattr(wanted, 'openssl_profile', None)!r})"
        )

    # The whole trick depends on RTCDtlsTransport reading the module global.
    transport_init = getattr(rtcdtlstransport.RTCDtlsTransport, "__init__", None)
    code = getattr(transport_init, "__code__", None)
    if code is None or "SRTP_PROFILES" not in code.co_names:
        raise RuntimeError(
            "aiortc.RTCDtlsTransport.__init__ no longer reads the module-level "
            "SRTP_PROFILES list, so restricting it in place would silently "
            "stop working; audio would fail with Chime close 1006"
        )

    if profiles == [wanted]:
        logger.debug(
            "SRTP profile already restricted to %s", SRTP_PROFILE_NAME.decode()
        )
        return

    if wanted not in profiles:
        # pylibsrtp was built without it, or aiortc stopped offering it.
        raise RuntimeError(
            f"{SRTP_PROFILE_NAME.decode()} is not among aiortc's available SRTP "
            f"profiles {[p.openssl_profile for p in profiles]!r}; Chime's audio "
            "server cannot use anything else (check the libsrtp2 build)"
        )

    dropped = [
        p.openssl_profile.decode()
        for p in profiles
        if p.openssl_profile != SRTP_PROFILE_NAME
    ]
    profiles[:] = [wanted]

    # Verify, rather than trust, the mutation.
    if rtcdtlstransport.SRTP_PROFILES != [wanted] or len(profiles) != 1:
        raise RuntimeError(
            "failed to restrict aiortc.rtcdtlstransport.SRTP_PROFILES to "
            f"{SRTP_PROFILE_NAME.decode()}; it is now "
            f"{rtcdtlstransport.SRTP_PROFILES!r}"
        )
    logger.info(
        "restricted aiortc DTLS-SRTP profiles to %s (dropped %s) -- Chime "
        "cannot decrypt AES-GCM and would drop the attendee ~2.5s in",
        SRTP_PROFILE_NAME.decode(),
        ", ".join(dropped) or "nothing",
    )


def force_chime_audio_codec() -> None:
    """Make aiortc offer exactly Chime's Opus line (PT 111 + fmtp).

    aiortc's stock audio codec list is ``opus/48000/2`` on payload type **96**
    with no ``fmtp``, plus G722/PCMU/PCMA. The doc (step 3) specifies Opus,
    payload type 111, ``minptime=10;useinbandfec=1``, so
    ``aiortc.codecs.CODECS["audio"]`` is rewritten **in place** to that single
    entry: ``createOffer()`` reads the module-level dict, and so does
    ``setRemoteDescription()`` when it intersects our codecs with Chime's
    answer.

    This is a process-wide change to aiortc (only one Opus profile remains
    usable afterwards), which is fine for this package -- a huddle leg wants
    nothing else -- but it is a real side effect, so it lives in its own named
    function. Idempotent.

    :raises RuntimeError: if ``aiortc.codecs.CODECS`` has changed shape.
    """
    try:
        from aiortc import codecs as aiortc_codecs
        from aiortc.rtcrtpparameters import RTCRtpCodecParameters
    except ImportError as exc:  # pragma: no cover - aiortc is a hard dep
        raise RuntimeError(f"cannot import aiortc codec tables: {exc}") from exc

    table = getattr(aiortc_codecs, "CODECS", None)
    if not isinstance(table, dict) or not isinstance(table.get("audio"), list):
        raise RuntimeError(  # noqa: TRY004 - one error type for "aiortc changed"
            "aiortc.codecs.CODECS['audio'] is not a list; cannot pin Opus to "
            f"payload type {OPUS_PAYLOAD_TYPE}"
        )

    opus = RTCRtpCodecParameters(
        mimeType="audio/opus",
        clockRate=OPUS_SAMPLE_RATE,
        channels=OPUS_CHANNELS,
        payloadType=OPUS_PAYLOAD_TYPE,
        parameters=dict(OPUS_FMTP),
    )
    if table["audio"] == [opus]:
        logger.debug("audio codec list already pinned to Opus PT %d", OPUS_PAYLOAD_TYPE)
        return
    table["audio"][:] = [opus]
    logger.info(
        "pinned aiortc audio codecs to opus/%d/%d PT %d (%s)",
        OPUS_SAMPLE_RATE,
        OPUS_CHANNELS,
        OPUS_PAYLOAD_TYPE,
        ";".join(f"{k}={v}" for k, v in OPUS_FMTP.items()),
    )


def _prefer_h264(transceiver: RTCRtpTransceiver) -> None:
    """Put H.264 first in a video transceiver's codec preferences.

    A share arrives as H.264 (doc step 4), and aiortc offers VP8 first. The
    payload numbers are aiortc's own (99 and 101 for the two H.264 profiles,
    with 100/102 as their RTX); the doc's offer happened to use 102 for H.264
    because werift numbers differently. That mismatch is harmless -- an
    answerer must use the payload types from the offer.
    """
    from aiortc import codecs as aiortc_codecs

    try:
        capabilities = aiortc_codecs.get_capabilities("video").codecs
    except Exception as exc:  # noqa: BLE001 - codec preference is best-effort
        logger.warning("cannot read aiortc video capabilities: %s", exc)
        return

    def kind(cap: RTCRtpCodecCapability) -> str:
        return cap.mimeType.lower()

    h264 = [c for c in capabilities if kind(c) == "video/h264"]
    rtx = [c for c in capabilities if kind(c) == "video/rtx"]
    others = [c for c in capabilities if kind(c) not in ("video/h264", "video/rtx")]
    if not h264:
        logger.warning(
            "aiortc offers no H.264 video codec; a Slack screen share (H.264) "
            "would not be receivable"
        )
        return
    try:
        transceiver.setCodecPreferences([*h264, *others, *rtx])
    except Exception as exc:  # noqa: BLE001 - codec preference is best-effort
        logger.warning("could not prefer H.264 on the video transceiver: %s", exc)


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #


class WallClockPacer:
    """A drift-free schedule of fixed-interval deadlines.

    ``docs/huddles-and-chime.md`` step 3: *"Pace audio off the wall clock. A
    20 ms timer sharing an event loop with heavy work fires late and the audio
    comes out crackly."* Sleeping ``interval`` seconds per frame accumulates
    every wakeup's lateness; instead the deadlines are accumulated
    (``start + n * interval``) and only the *remaining* time is slept, so a
    late wakeup is corrected by the next, shorter sleep.

    If the schedule falls further behind than ``max_slip`` seconds (the event
    loop was blocked for a long time, the process was suspended, …) catching
    up would mean a burst of back-to-back packets, which is worse than a gap:
    the schedule is resynchronised to "now" instead and :attr:`resyncs` is
    incremented.

    Pure and clock-injectable, so the pacing can be tested without sleeping.
    """

    __slots__ = ("_deadline", "interval", "max_slip", "resyncs")

    def __init__(self, interval: float = AUDIO_PTIME, *, max_slip: float = 0.5) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.max_slip = max_slip
        self.resyncs = 0
        self._deadline: float | None = None

    @property
    def deadline(self) -> float | None:
        """The absolute time the next frame is due, or ``None`` before start."""
        return self._deadline

    def reset(self) -> None:
        self._deadline = None
        self.resyncs = 0

    def next_delay(self, now: float) -> float:
        """Seconds to sleep before emitting the next frame; advances the schedule.

        The first call returns ``0.0`` (emit immediately) and anchors the grid
        at ``now``. Later calls return ``max(0.0, deadline - now)``, where
        ``deadline`` is ``anchor + n * interval`` -- never ``interval`` itself,
        which is what drifts.
        """
        if self._deadline is None:
            self._deadline = now + self.interval
            return 0.0
        delay = self._deadline - now
        if delay < -self.max_slip:
            # Debug, not a warning: this leg exists only so Chime reports us in
            # AUDIO_STREAM_ID_INFO and other clients accept our drawings. The
            # payload is near-silence, so a late frame is inaudible. Pacing
            # would matter if this were ever used to play real audio (the doc's
            # "crackly" note is about exactly that), which is why the resync is
            # kept -- but it is not worth a warning while we are drawing.
            logger.debug(
                "audio pacing fell %.0f ms behind; resynchronising the 20 ms grid",
                -delay * 1000,
            )
            self.resyncs += 1
            self._deadline = now + self.interval
            return 0.0
        self._deadline += self.interval
        return max(0.0, delay)


# --------------------------------------------------------------------------- #
# The silence source
# --------------------------------------------------------------------------- #


class SilenceAudioTrack(MediaStreamTrack):
    """20 ms frames of near-silence, paced off the wall clock.

    The doc (step 3) says *"send 20 ms Opus packets (pre-encoded near-silence
    is enough)"*. aiortc owns the Opus encoder, so this track produces the PCM
    and lets aiortc encode it; a constant +/-1 LSB signal is inaudible and
    compresses to 3-byte Opus packets, which is all Chime needs to mark the
    attendee present. ``amplitude=0`` gives exact silence.

    aiortc's own :class:`~aiortc.mediastreams.AudioStreamTrack` is 8 kHz mono;
    this one is 48 kHz stereo to match ``opus/48000/2``, and uses
    :class:`WallClockPacer` plus a monotonic clock.

    The clock and the sleep function are injectable so pacing can be tested
    without real time passing.
    """

    kind = "audio"

    def __init__(
        self,
        *,
        amplitude: int = 1,
        sample_rate: int = OPUS_SAMPLE_RATE,
        channels: int = OPUS_CHANNELS,
        ptime: float = AUDIO_PTIME,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        super().__init__()
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2 (aiortc's Opus encoder)")
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_frame = round(sample_rate * ptime)
        self.pacer = WallClockPacer(ptime)
        self.frames_sent = 0
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._layout = "stereo" if channels == 2 else "mono"
        self._time_base = fractions.Fraction(1, sample_rate)
        self._pts = 0
        self._payload = self._make_payload()

    def _make_payload(self) -> bytes:
        """Interleaved s16 samples alternating ``+amplitude`` / ``-amplitude``."""
        value = max(0, min(32767, abs(self.amplitude)))
        if value == 0:
            return bytes(self.samples_per_frame * self.channels * 2)
        positive = value.to_bytes(2, "little", signed=True) * self.channels
        negative = (-value).to_bytes(2, "little", signed=True) * self.channels
        pair = positive + negative
        frames = self.samples_per_frame // 2
        payload = pair * frames
        if self.samples_per_frame % 2:  # pragma: no cover - 960 is even
            payload += positive
        return payload

    def make_frame(self) -> av.AudioFrame:
        """Build the next 20 ms frame (no pacing, no state beyond ``pts``)."""
        frame = av.AudioFrame(
            format="s16", layout=self._layout, samples=self.samples_per_frame
        )
        frame.planes[0].update(self._payload)
        frame.sample_rate = self.sample_rate
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += self.samples_per_frame
        return frame

    async def recv(self) -> av.AudioFrame:
        """Sleep until the next 20 ms deadline, then return a frame.

        :raises MediaStreamError: once :meth:`stop` has been called, which is
            how aiortc's sender learns to shut down.
        """
        if self.readyState != "live":
            raise MediaStreamError
        delay = self.pacer.next_delay(self._clock())
        if delay > 0:
            await self._sleep(delay)
        if self.readyState != "live":
            raise MediaStreamError
        self.frames_sent += 1
        return self.make_frame()


# --------------------------------------------------------------------------- #
# The media leg
# --------------------------------------------------------------------------- #


class AudioLeg:
    """The audio/video peer connection that makes the bot a present attendee.

    Usage, matching ``docs/huddles-and-chime.md`` step 3::

        leg = AudioLeg()
        offer = await leg.create_offer()           # ICE already gathered
        answer = await signaling.subscribe(sdp_offer=offer, audio_host=...)
        await leg.accept_answer(answer)
        ...
        await leg.close()

    The offer has a ``sendrecv`` audio section (Opus PT 111,
    ``minptime=10;useinbandfec=1``) and a ``recvonly`` video section --
    ``SUBSCRIBE_ACK`` fails with "failed to initialize video session" without
    the latter, even when no video is wanted.

    Incoming media (the huddle's mixed Opus audio, and a share's H.264 video
    once its ``streamId`` is in ``receiveStreamIds``) is handed to callbacks
    registered with :meth:`on_track`. **A callback owns the track and must
    consume it**: aiortc's receiver decodes into an unbounded queue, so a
    track nobody reads grows in memory. Tracks that arrive with no callback
    registered are therefore drained and discarded, with a warning.
    """

    def __init__(
        self,
        *,
        muted: bool = True,
        amplitude: int = 1,
        ice_servers: Sequence[RTCIceServer] | None = None,
        relay_only: bool | None = None,
        video_direction: str = "inactive",
        sdp_origin_username: str | None = SDP_ORIGIN_USERNAME,
    ) -> None:
        """
        :param muted: recorded for the caller to pass as ``audio_muted`` in
            ``SUBSCRIBE``. It does **not** stop packets: the whole point of
            this leg is that audio keeps flowing, muted or not. It only
            selects exact silence over the +/-1 LSB dither.
        :param amplitude: sample amplitude of the near-silence, in s16 LSBs.
        :param ice_servers: ICE servers. For a real huddle these **must** be
            Chime's own TURN servers from the ``JOIN_ACK`` -- build them with
            :func:`ice_servers_from_turn_credentials`. ``None`` keeps aiortc's
            default (a Google STUN server), which Chime's media server will
            not accept. Pass ``[]`` for host candidates only, which is useful
            offline and in tests.
        :param relay_only: restrict ICE to relay candidates. Defaults to
            ``True`` whenever ``ice_servers`` contains a TURN server, because
            that is what the real client does -- see below.
        :param sdp_origin_username: rewrite the offer's ``o=`` username to
            this (the Chime JS SDK sends ``mozilla-chrome``); ``None`` leaves
            aiortc's ``o=-`` alone.

        **Chime media only travels over Chime's TURN relay.**
        ``CreatePeerConnectionTask`` in amazon-chime-sdk-js builds its
        ``RTCConfiguration`` from the ``JOIN_ACK`` TURN credentials *and* sets
        ``iceTransportPolicy: 'relay'``. An attendee that offers host or
        ``srflx`` candidates instead gets no media through: the symptom is the
        signaling websocket closing with **1006** a couple of seconds in,
        which looks identical to the SRTP failure and is not.
        """
        # Must happen before any RTCDtlsTransport exists, and it is cheap, so
        # it is done here rather than left to the caller to remember.
        force_srtp_profile()
        force_chime_audio_codec()
        force_chime_dtls_ciphers()

        self.muted = muted
        self.sdp_origin_username = sdp_origin_username

        if relay_only is None:
            relay_only = bool(ice_servers) and any(
                _is_turn_server(server) for server in ice_servers
            )
        self.relay_only = relay_only

        # `max-bundle` is what the real client uses unconditionally
        # (`DefaultBrowserBehavior.requiresBundlePolicy()` returns
        # 'max-bundle'), and it matters: with the default balanced policy
        # aiortc gives the audio and video m-lines separate ICE transports, so
        # it makes **two** TURN allocations and advertises a *different* relay
        # candidate in each m-section, then tears one down once BUNDLE is
        # negotiated. Chime answers with everything bundled onto mid 0, and the
        # DTLS handshake then goes nowhere. One bundle, one transport, one
        # allocation, one set of candidates.
        configuration = RTCConfiguration(
            iceServers=list(ice_servers) if ice_servers is not None else None,
            iceTransportPolicy=(
                RTCIceTransportPolicy.RELAY
                if relay_only
                else RTCIceTransportPolicy.ALL
            ),
            bundlePolicy=RTCBundlePolicy.MAX_BUNDLE,
        )
        logger.info(
            "audio leg ICE: %s server(s), transport policy %s, bundle policy %s",
            len(configuration.iceServers) if configuration.iceServers else "default",
            configuration.iceTransportPolicy.value,
            configuration.bundlePolicy.value,
        )
        self._pc = RTCPeerConnection(configuration=configuration)
        self.track = SilenceAudioTrack(amplitude=0 if muted else amplitude)
        self._audio_transceiver = self._pc.addTransceiver(
            self.track, direction="sendrecv"
        )
        # The video section must exist (without one, SUBSCRIBE_ACK fails with
        # "failed to initialize video session"), but its *direction* has to
        # agree with the `receiveStreamIds` we send alongside it. Chime's
        # backend requires those to line up positionally with the video media
        # sections, "with a zero for any video send or inactive section"
        # (SubscribeAndReceiveSubscribeAckTask.fixUpSubscriptionOrder in
        # amazon-chime-sdk-js). A `recvonly` section must therefore carry a
        # real stream id, never 0 -- so a client that subscribes to nothing
        # yet sends an **inactive** section, which is what the real client
        # does when there is no camera. To receive a share, add a further
        # recvonly transceiver and pass [0, <streamId>].
        self._video_transceiver = self._pc.addTransceiver(
            "video", direction=video_direction
        )
        _prefer_h264(self._video_transceiver)

        self._track_callbacks: list[TrackCallback] = []
        self._tasks: set[asyncio.Task[Any]] = set()
        self._pli_task: asyncio.Task[None] | None = None
        self._offer_created = False
        self._answered = False
        self._closed = False
        self.audio_track: MediaStreamTrack | None = None
        self.video_track: MediaStreamTrack | None = None

        @self._pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            self._handle_track(track)

        # Media bring-up is where a Chime attendee most often dies, and the
        # symptom shows up somewhere else entirely (the signaling websocket
        # closing 1006). Log every state transition and wake anyone waiting in
        # wait_until_connected() so the failure is reported where it happens.
        self._connected = asyncio.Event()
        self._media_failed: str | None = None

        @self._pc.on("connectionstatechange")
        def _on_state() -> None:
            state = self._pc.connectionState
            logger.info("audio leg connection state: %s", state)
            if state == "connected":
                self._connected.set()
                self._log_selected_candidates()
            elif state in ("failed", "closed"):
                if state == "failed":
                    self._media_failed = (
                        "the WebRTC connection failed (ICE or DTLS did not "
                        "complete)"
                    )
                self._connected.set()

        @self._pc.on("iceconnectionstatechange")
        def _on_ice_state() -> None:
            logger.info(
                "audio leg ICE connection state: %s", self._pc.iceConnectionState
            )
            if self._pc.iceConnectionState == "failed":
                self._media_failed = (
                    "ICE failed: no usable candidate pair with Chime's media "
                    "server. With relay-only transport this means the TURN "
                    "allocation on Chime's servers did not succeed"
                )
                self._connected.set()

        @self._pc.on("icegatheringstatechange")
        def _on_gathering() -> None:
            logger.debug(
                "audio leg ICE gathering state: %s", self._pc.iceGatheringState
            )

    # -- properties -------------------------------------------------------- #

    @property
    def pc(self) -> RTCPeerConnection:
        """The underlying peer connection (for stats, state, debugging)."""
        return self._pc

    @property
    def is_running(self) -> bool:
        """True once the answer is applied and the connection is still alive.

        False before :meth:`accept_answer`, after :meth:`close`, and once the
        peer connection reports ``closed`` or ``failed``.
        """
        if self._closed or not self._answered:
            return False
        return self._pc.connectionState not in ("closed", "failed")

    @property
    def connection_state(self) -> str:
        return self._pc.connectionState

    def _log_selected_candidates(self) -> None:
        """Log the chosen candidate pair and negotiated SRTP profile.

        Both are the first things worth knowing when Chime drops an attendee,
        and neither is otherwise visible without aiortc debug logging.
        """
        try:
            transceiver = self._pc.getTransceivers()[0]
            ice = transceiver.receiver.transport.transport
            pair = getattr(ice, "_connection", None)
            nominated = getattr(pair, "nominated_keys", None) or getattr(
                pair, "_nominated", None
            )
            local = [
                f"{c.type}:{c.host}:{c.port}" for c in ice.getLocalCandidates()[:4]
            ]
            remote = [
                f"{c.type}:{c.host}:{c.port}" for c in ice.getRemoteCandidates()[:4]
            ]
            logger.info(
                "audio leg connected — local candidates %s, remote candidates %s%s",
                local or "none",
                remote or "none",
                f", nominated {nominated}" if nominated else "",
            )
            dtls = transceiver.receiver.transport
            profile = getattr(dtls, "_srtp_profile", None)
            if profile is not None:
                logger.info(
                    "audio leg DTLS-SRTP profile: %s",
                    getattr(profile, "openssl_profile", profile),
                )
        except Exception:  # diagnostics must never break the session
            logger.debug("could not log the selected candidates", exc_info=True)

    async def wait_until_connected(self, timeout: float = 20.0) -> None:
        """Wait for ICE and DTLS to come up, or raise with a usable reason.

        Without this the caller finds out only indirectly: Chime closes the
        *signaling* websocket with 1006 a second or two after a media leg that
        never connected, which reads as an SRTP or rate-limit problem and is
        neither.
        """
        if self._pc.connectionState == "connected":
            return
        try:
            await asyncio.wait_for(self._connected.wait(), timeout)
        except TimeoutError:
            raise MediaConnectionError(
                f"the audio leg did not connect within {timeout:g}s "
                f"(connection state {self._pc.connectionState!r}, ICE state "
                f"{self._pc.iceConnectionState!r}). Chime media only traverses "
                "Chime's own TURN relay, so check that JOIN_ACK supplied TURN "
                "credentials and that they reached the peer connection."
            ) from None
        if self._media_failed is not None:
            raise MediaConnectionError(self._media_failed)
        if self._pc.connectionState != "connected":
            raise MediaConnectionError(
                f"the audio leg stopped before connecting "
                f"(state {self._pc.connectionState!r})"
            )

    # -- negotiation ------------------------------------------------------- #

    async def create_offer(self) -> str:
        """Create the offer, gather ICE, and return the complete SDP.

        ICE gathering happens inside ``setLocalDescription()``, so the string
        returned already carries every candidate and ``a=end-of-candidates``:
        the caller can hand it straight to ``SUBSCRIBE`` (Chime has no
        trickle-ICE path here).
        """
        if self._closed:
            raise RuntimeError("AudioLeg is closed")
        if self._offer_created:
            raise RuntimeError("create_offer() was already called")
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        self._offer_created = True
        local = self._pc.localDescription
        sdp = local.sdp if local is not None else offer.sdp
        sdp = self._rewrite_origin(sdp)
        # Chime rejects aiortc's three-fingerprint offer with a fatal
        # unknown_ca during certificate verification.
        sdp = keep_single_fingerprint(sdp)

        # The SDK refuses to subscribe without at least one relay candidate
        # (FinishGatheringICECandidatesTask: "no ice candidates were
        # gathered"), and under relay-only ICE aioice gathers nothing at all if
        # the TURN allocation failed -- which would otherwise show up much
        # later as an unexplained media timeout.
        relay = [
            line
            for line in sdp.splitlines()
            if line.startswith("a=candidate:") and " typ relay " in line
        ]
        if self.relay_only and not relay:
            raise MediaConnectionError(
                "no relay ICE candidates were gathered, so Chime's media "
                "server is unreachable. The TURN allocation against Chime's "
                "ICE servers failed -- check the JOIN_ACK TURN credentials."
            )
        logger.info(
            "audio leg offer ready (ICE gathering %s, %d relay candidate(s), "
            "%d bytes)",
            self._pc.iceGatheringState,
            len(relay),
            len(sdp),
        )
        logger.debug("offer SDP:\n%s", sdp)
        return sdp

    def _rewrite_origin(self, sdp: str) -> str:
        """Mimic the Chime SDK's ``o=-`` -> ``o=mozilla-chrome`` rewrite (step 3)."""
        if not self.sdp_origin_username:
            return sdp
        lines = sdp.split("\n")
        for index, line in enumerate(lines):
            if line.startswith("o="):
                fields = line[2:].split(" ")
                if fields:
                    fields[0] = self.sdp_origin_username
                    lines[index] = "o=" + " ".join(fields)
                break
        return "\n".join(lines)

    async def accept_answer(self, sdp_answer: str) -> None:
        """Apply the SDP answer from ``SUBSCRIBE_ACK``.

        DTLS and SRTP come up afterwards; aiortc logs the negotiated profile
        itself ("DTLS handshake negotiated …"), which is the place to look if
        Chime starts dropping the attendee again.
        """
        if self._closed:
            raise RuntimeError("AudioLeg is closed")
        if not self._offer_created:
            raise RuntimeError("accept_answer() before create_offer()")
        # Chime puts its candidates only on the primary bundled m-section,
        # which aiortc then loses under max-bundle. See
        # propagate_bundle_candidates().
        sdp_answer = propagate_bundle_candidates(sdp_answer)
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp_answer, type="answer")
        )
        self._answered = True
        self._log_negotiated_audio()

    def _log_negotiated_audio(self) -> None:
        codecs = getattr(self._audio_transceiver, "_codecs", [])
        described = [f"{c.name}/{c.clockRate} PT {c.payloadType}" for c in codecs]
        logger.info("audio leg negotiated: %s", ", ".join(described) or "nothing")
        if not any(
            c.name.lower() == "opus" and c.payloadType == OPUS_PAYLOAD_TYPE
            for c in codecs
        ):
            logger.warning(
                "the answer did not keep Opus on payload type %d; Chime may not "
                "accept our packets",
                OPUS_PAYLOAD_TYPE,
            )

    # -- receiving --------------------------------------------------------- #

    def on_track(self, cb: TrackCallback) -> Callable[[], None]:
        """Register a callback for each incoming track; returns an unsubscribe.

        The callback gets the :class:`~aiortc.mediastreams.MediaStreamTrack`
        (``track.kind`` is ``"audio"`` for the huddle's mixed Opus and
        ``"video"`` for a subscribed share's H.264) and must consume it with
        ``await track.recv()``. Callbacks may be async. Register before
        :meth:`accept_answer`, or the track may already be being drained.
        """
        self._track_callbacks.append(cb)

        def unsubscribe() -> None:
            try:
                self._track_callbacks.remove(cb)
            except ValueError:
                pass

        return unsubscribe

    def _handle_track(self, track: MediaStreamTrack) -> None:
        logger.info("audio leg received a %s track", track.kind)
        if track.kind == "audio":
            self.audio_track = track
        elif track.kind == "video":
            self.video_track = track
        if not self._track_callbacks:
            # Expected for a draw-only attendee: it has to be *present* (so the
            # media leg is mandatory) but has no use for the huddle's audio or
            # anyone's video. Draining keeps aiortc's decode queue from
            # growing, so this is handled, not a problem -- hence debug.
            logger.debug(
                "no on_track callback registered; draining the incoming %s "
                "track so aiortc's decode queue does not grow",
                track.kind,
            )
            self._spawn(self._drain(track))
            return
        for cb in list(self._track_callbacks):
            try:
                result = cb(track)
            except Exception:
                logger.exception("on_track callback failed")
                continue
            if inspect.isawaitable(result):
                self._spawn(result)

    async def _drain(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                await track.recv()
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:  # pragma: no cover - defensive
            logger.exception("draining the %s track failed", track.kind)

    async def send_video_pli(self) -> bool:
        """Send one RTCP PLI for each SSRC seen on the video receiver.

        The doc (step 4) says to send a PLI every couple of seconds so a
        decoder gets keyframes. aiortc has no public API for this, so this
        calls the receiver's private ``_send_rtcp_pli(media_ssrc)``; the SSRCs
        come from ``getSynchronizationSources()``, which only knows SSRCs seen
        in the last 10 seconds. **Limitations, not papered over:**

        * before any video RTP has arrived there is no SSRC to address and
          this returns ``False``;
        * an RTX SSRC on the same receiver would also get a (harmless) PLI;
        * ``_send_rtcp_pli`` is private API and may change with aiortc. It is
          absent/renamed -> a warning and ``False``, never a crash.

        aiortc also sends PLIs by itself when its own decoder reports lost
        picture data, so this is an addition, not the only source of PLIs.

        :returns: whether at least one PLI was sent.
        """
        receiver = self._video_receiver()
        if receiver is None or self._closed:
            return False
        send_pli = getattr(receiver, "_send_rtcp_pli", None)
        if send_pli is None:
            logger.warning(
                "this aiortc has no RTCRtpReceiver._send_rtcp_pli; cannot "
                "request keyframes manually"
            )
            return False
        sources = receiver.getSynchronizationSources()
        if not sources:
            logger.debug("no video SSRC seen yet; not sending a PLI")
            return False
        sent = False
        for source in sources:
            try:
                await send_pli(source.source)
                sent = True
            except Exception as exc:  # noqa: BLE001 - a lost PLI is not fatal
                logger.warning(
                    "sending RTCP PLI for ssrc %s failed: %s", source.source, exc
                )
        return sent

    def start_video_pli(self, interval: float = 2.0) -> None:
        """Send an RTCP PLI every ``interval`` seconds (doc step 4)."""
        if self._closed:
            raise RuntimeError("AudioLeg is closed")
        if self._pli_task is not None and not self._pli_task.done():
            return
        self._pli_task = asyncio.ensure_future(self._pli_loop(interval))

    async def _pli_loop(self, interval: float) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                await self.send_video_pli()
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive
            logger.exception("the periodic PLI loop died")

    async def stop_video_pli(self) -> None:
        """Stop the periodic PLI task, if any. Idempotent."""
        task, self._pli_task = self._pli_task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("the periodic PLI task failed while stopping")

    def _video_receiver(self) -> RTCRtpReceiver | None:
        for transceiver in self._pc.getTransceivers():
            if transceiver.kind == "video":
                return transceiver.receiver
        return None

    # -- teardown ---------------------------------------------------------- #

    async def close(self) -> None:
        """Close the peer connection and stop the silence track.

        Idempotent, and safe to call after a failed :meth:`create_offer` or
        :meth:`accept_answer` -- errors from aiortc's teardown are logged, not
        raised, so a caller can always use this in a ``finally``.
        """
        if self._closed:
            return
        self._closed = True
        self._answered = False
        await self.stop_video_pli()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            self._tasks.clear()
        try:
            self.track.stop()
        except Exception:  # pragma: no cover - defensive
            logger.exception("stopping the silence track failed")
        try:
            await self._pc.close()
        except Exception as exc:  # noqa: BLE001 - close() must be safe in a finally
            logger.warning("closing the audio leg's peer connection failed: %s", exc)
        logger.info("audio leg closed after %d frames", self.track.frames_sent)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


__all__ = [
    "AUDIO_PTIME",
    "OPUS_FMTP",
    "OPUS_PAYLOAD_TYPE",
    "SAMPLES_PER_FRAME",
    "SRTP_PROFILE_NAME",
    "AudioLeg",
    "SilenceAudioTrack",
    "WallClockPacer",
    "force_chime_audio_codec",
    "force_srtp_profile",
]
