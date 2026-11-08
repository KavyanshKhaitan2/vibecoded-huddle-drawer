"""Tests for the required media leg (``huddle_drawings.audio``).

No network and no real Chime: the peer connection is built with no ICE
servers, so only host candidates are gathered, and nothing is ever sent.
Pacing is tested through an injected clock rather than by sleeping.
"""

from __future__ import annotations

import asyncio
import fractions
from typing import Any

import pytest
import pytest_asyncio
from aiortc import codecs as aiortc_codecs
from aiortc import rtcdtlstransport
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from huddle_drawings.audio import (
    AUDIO_PTIME,
    OPUS_PAYLOAD_TYPE,
    SAMPLES_PER_FRAME,
    AudioLeg,
    SilenceAudioTrack,
    WallClockPacer,
    force_chime_audio_codec,
    force_srtp_profile,
)


@pytest.fixture(autouse=True)
def restore_aiortc_globals() -> Any:
    """Undo the in-place mutations the module makes to aiortc's globals."""
    srtp = list(rtcdtlstransport.SRTP_PROFILES)
    audio_codecs = list(aiortc_codecs.CODECS["audio"])
    try:
        yield
    finally:
        rtcdtlstransport.SRTP_PROFILES[:] = srtp
        aiortc_codecs.CODECS["audio"][:] = audio_codecs


# --------------------------------------------------------------------------- #
# force_srtp_profile
# --------------------------------------------------------------------------- #


def test_force_srtp_profile_leaves_only_cm_sha1_80() -> None:
    before = list(rtcdtlstransport.SRTP_PROFILES)
    assert len(before) > 1, "expected aiortc to offer AES-GCM profiles too"

    force_srtp_profile()

    assert rtcdtlstransport.SRTP_PROFILES == [rtcdtlstransport.SRTP_AES128_CM_SHA1_80]
    assert [p.openssl_profile for p in rtcdtlstransport.SRTP_PROFILES] == [
        b"SRTP_AES128_CM_SHA1_80"
    ]


def test_force_srtp_profile_mutates_in_place_and_is_idempotent() -> None:
    # A reference taken before the call (as aiortc's own transports do) must
    # see the restriction.
    held_reference = rtcdtlstransport.SRTP_PROFILES

    force_srtp_profile()
    force_srtp_profile()
    force_srtp_profile()

    assert held_reference is rtcdtlstransport.SRTP_PROFILES
    assert held_reference == [rtcdtlstransport.SRTP_AES128_CM_SHA1_80]


def test_force_srtp_profile_rejects_changed_internals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rtcdtlstransport, "SRTP_PROFILES", (rtcdtlstransport.SRTP_AES128_CM_SHA1_80,)
    )
    with pytest.raises(RuntimeError, match="no longer a list"):
        force_srtp_profile()


def test_force_srtp_profile_rejects_missing_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(rtcdtlstransport, "SRTP_AES128_CM_SHA1_80")
    with pytest.raises(RuntimeError, match="no longer exposes"):
        force_srtp_profile()


def test_force_srtp_profile_rejects_unavailable_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rtcdtlstransport,
        "SRTP_PROFILES",
        [rtcdtlstransport.SRTP_AEAD_AES_128_GCM],
    )
    with pytest.raises(RuntimeError, match="not among aiortc's available"):
        force_srtp_profile()


def test_force_srtp_profile_rejects_detached_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If RTCDtlsTransport stops reading the module global, mutating is useless."""

    class Detached:
        def __init__(self) -> None:  # does not mention SRTP_PROFILES
            pass

    monkeypatch.setattr(rtcdtlstransport, "RTCDtlsTransport", Detached)
    with pytest.raises(RuntimeError, match="no longer reads the module-level"):
        force_srtp_profile()


def test_force_chime_audio_codec_pins_opus_111() -> None:
    held_reference = aiortc_codecs.CODECS["audio"]

    force_chime_audio_codec()
    force_chime_audio_codec()

    assert held_reference is aiortc_codecs.CODECS["audio"]
    assert len(held_reference) == 1
    codec = held_reference[0]
    assert codec.name.lower() == "opus"
    assert codec.payloadType == OPUS_PAYLOAD_TYPE
    assert codec.clockRate == 48000
    assert codec.channels == 2
    assert codec.parameters == {"minptime": 10, "useinbandfec": 1}


# --------------------------------------------------------------------------- #
# The offer
# --------------------------------------------------------------------------- #


def sdp_sections(sdp: str) -> dict[str, list[str]]:
    """Split an SDP into ``{"audio": [...lines], "video": [...]}`` m-sections."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            kind = line[2:].split(" ")[0]
            current = sections.setdefault(kind, [])
        if current is not None:
            current.append(line)
    return sections


@pytest_asyncio.fixture
async def leg() -> Any:
    leg = AudioLeg(ice_servers=[])
    try:
        yield leg
    finally:
        await leg.close()


@pytest.mark.asyncio
async def test_offer_has_sendrecv_opus_audio(leg: AudioLeg) -> None:
    sdp = await leg.create_offer()
    audio = sdp_sections(sdp)["audio"]

    assert audio[0].startswith("m=audio ")
    assert str(OPUS_PAYLOAD_TYPE) in audio[0].split(" ")[3:]
    assert "a=sendrecv" in audio
    assert f"a=rtpmap:{OPUS_PAYLOAD_TYPE} opus/48000/2" in audio
    assert f"a=fmtp:{OPUS_PAYLOAD_TYPE} minptime=10;useinbandfec=1" in audio


@pytest.mark.asyncio
async def test_offer_has_inactive_h264_video(leg: AudioLeg) -> None:
    """A video m-section is mandatory, and it must be *inactive* by default.

    Without any video m-section ``SUBSCRIBE_ACK`` fails with "failed to
    initialize video session" (doc step 3). But the direction has to agree
    with the ``receiveStreamIds`` sent alongside it: Chime requires those to
    line up positionally with the video sections, "with a zero for any video
    send or inactive section"
    (``SubscribeAndReceiveSubscribeAckTask.fixUpSubscriptionOrder`` in
    amazon-chime-sdk-js). Since we subscribe to nothing by default we send
    ``[0]``, so the section must be ``inactive`` -- a ``recvonly`` section
    would have to carry a real stream id.
    """
    sdp = await leg.create_offer()
    sections = sdp_sections(sdp)
    assert "video" in sections, "a video section is mandatory"
    video = sections["video"]

    assert "a=inactive" in video
    assert "a=recvonly" not in video
    h264_rtpmaps = [
        line for line in video if line.startswith("a=rtpmap:") and "H264/90000" in line
    ]
    assert h264_rtpmaps, "a share arrives as H.264, so the offer must receive it"

    # H.264 is preferred: the first format in the m-line is an H.264 PT.
    first_pt = video[0].split(" ")[3]
    assert f"a=rtpmap:{first_pt} H264/90000" in video


@pytest.mark.asyncio
async def test_offer_has_completed_ice_gathering(leg: AudioLeg) -> None:
    sdp = await leg.create_offer()

    assert leg.pc.iceGatheringState == "complete"
    assert "a=end-of-candidates" in sdp
    for kind, lines in sdp_sections(sdp).items():
        candidates = [line for line in lines if line.startswith("a=candidate:")]
        assert candidates, f"no ICE candidates in the {kind} section"
        assert any("typ host" in line for line in candidates)


@pytest.mark.asyncio
async def test_offer_origin_is_rewritten_like_the_chime_sdk(leg: AudioLeg) -> None:
    sdp = await leg.create_offer()
    origin = next(line for line in sdp.split("\n") if line.startswith("o="))
    assert origin.startswith("o=mozilla-chrome ")
    assert len(origin.split(" ")) == 6


@pytest.mark.asyncio
async def test_offer_origin_can_be_left_alone() -> None:
    leg = AudioLeg(ice_servers=[], sdp_origin_username=None)
    try:
        sdp = await leg.create_offer()
    finally:
        await leg.close()
    assert any(line.startswith("o=- ") for line in sdp.split("\n"))


@pytest.mark.asyncio
async def test_create_offer_is_single_shot(leg: AudioLeg) -> None:
    await leg.create_offer()
    with pytest.raises(RuntimeError, match="already called"):
        await leg.create_offer()


@pytest.mark.asyncio
async def test_accept_answer_requires_an_offer(leg: AudioLeg) -> None:
    with pytest.raises(RuntimeError, match="before create_offer"):
        await leg.accept_answer("v=0\n")


@pytest.mark.asyncio
async def test_accept_answer_marks_the_leg_running(
    leg: AudioLeg, caplog: pytest.LogCaptureFixture
) -> None:
    """``setRemoteDescription`` is stubbed: no DTLS, no Chime, no network."""
    applied: list[Any] = []

    async def fake_set_remote_description(description: Any) -> None:
        applied.append(description)

    leg.pc.setRemoteDescription = fake_set_remote_description  # type: ignore[method-assign]
    await leg.create_offer()
    with caplog.at_level("INFO", logger="huddle_drawings.audio"):
        await leg.accept_answer("v=0\r\nanswer-from-chime\r\n")

    assert applied[0].type == "answer"
    assert applied[0].sdp.startswith("v=0")
    assert leg.is_running is True
    messages = [record.getMessage() for record in caplog.records]
    assert any("negotiated: opus/48000 PT 111" in message for message in messages)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]

    await leg.close()
    assert leg.is_running is False


@pytest.mark.asyncio
async def test_is_running_and_close_is_idempotent(leg: AudioLeg) -> None:
    assert leg.is_running is False  # nothing negotiated yet
    await leg.create_offer()
    assert leg.is_running is False  # still no answer

    await leg.close()
    assert leg.is_running is False
    await leg.close()  # idempotent
    await leg.close()
    assert leg.pc.connectionState == "closed"


@pytest.mark.asyncio
async def test_close_is_safe_before_anything_happened() -> None:
    leg = AudioLeg(ice_servers=[])
    await leg.close()
    assert leg.is_running is False
    with pytest.raises(RuntimeError, match="closed"):
        await leg.create_offer()


@pytest.mark.asyncio
async def test_close_survives_a_broken_peer_connection(leg: AudioLeg) -> None:
    async def boom() -> None:
        raise ConnectionError("transport already gone")

    leg.pc.close = boom  # type: ignore[method-assign]
    await leg.close()  # logged, not raised
    assert leg.is_running is False


@pytest.mark.asyncio
async def test_on_track_dispatches_and_unsubscribes(leg: AudioLeg) -> None:
    class DummyTrack(MediaStreamTrack):
        kind = "video"

        async def recv(self) -> Any:  # pragma: no cover - never consumed
            raise MediaStreamError

    seen: list[MediaStreamTrack] = []

    async def callback(track: MediaStreamTrack) -> None:
        seen.append(track)

    unsubscribe = leg.on_track(callback)
    track = DummyTrack()
    leg._handle_track(track)
    await asyncio.sleep(0)
    assert seen == [track]
    assert leg.video_track is track

    unsubscribe()
    leg._handle_track(DummyTrack())
    await asyncio.sleep(0)
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_send_video_pli_without_media_returns_false(leg: AudioLeg) -> None:
    assert await leg.send_video_pli() is False


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #


def test_pacer_emits_the_first_frame_immediately() -> None:
    pacer = WallClockPacer(AUDIO_PTIME)
    assert pacer.deadline is None
    assert pacer.next_delay(100.0) == 0.0
    assert pacer.deadline == pytest.approx(100.02)


def test_pacer_does_not_drift_when_every_wakeup_is_late() -> None:
    """A naive sleep(0.02) would slip 5 ms per frame; the pacer must not."""
    pacer = WallClockPacer(AUDIO_PTIME)
    overhead = 0.005
    start = 1_000.0
    now = start
    emitted: list[float] = []

    for _ in range(100):
        delay = pacer.next_delay(now)
        assert delay >= 0.0
        now += delay + overhead  # sleep, then wake up late
        emitted.append(now)

    for index, when in enumerate(emitted):
        assert when == pytest.approx(start + index * AUDIO_PTIME + overhead, abs=1e-9)
    # 100 frames took 99 * 20 ms of media time, not 100 * 25 ms.
    assert emitted[-1] - emitted[0] == pytest.approx(99 * AUDIO_PTIME, abs=1e-9)
    assert pacer.resyncs == 0


def test_pacer_resyncs_instead_of_bursting_after_a_long_stall() -> None:
    pacer = WallClockPacer(AUDIO_PTIME, max_slip=0.5)
    pacer.next_delay(0.0)
    assert pacer.next_delay(0.02) == pytest.approx(0.0)

    # The event loop was blocked for two seconds.
    assert pacer.next_delay(2.0) == 0.0
    assert pacer.resyncs == 1
    assert pacer.deadline == pytest.approx(2.0 + AUDIO_PTIME)
    # Back on a fresh grid, no backlog burst.
    assert pacer.next_delay(2.0) == pytest.approx(AUDIO_PTIME)


def test_pacer_rejects_a_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        WallClockPacer(0.0)


# --------------------------------------------------------------------------- #
# The silence track
# --------------------------------------------------------------------------- #


class FakeClock:
    """A clock that only advances when the track "sleeps" (plus overhead)."""

    def __init__(self, start: float = 500.0, overhead: float = 0.004) -> None:
        self.now = start
        self.overhead = overhead
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay + self.overhead


def make_track(**kwargs: Any) -> tuple[SilenceAudioTrack, FakeClock]:
    clock = FakeClock()
    track = SilenceAudioTrack(clock=clock, sleep=clock.sleep, **kwargs)
    return track, clock


@pytest.mark.asyncio
async def test_silence_track_produces_20ms_48khz_stereo_frames() -> None:
    track, _clock = make_track()

    for index in range(5):
        frame = await track.recv()
        assert frame.samples == SAMPLES_PER_FRAME == 960
        assert frame.sample_rate == 48000
        assert frame.format.name == "s16"
        assert frame.layout.name == "stereo"
        assert frame.time_base == fractions.Fraction(1, 48000)
        assert frame.pts == index * 960
        assert frame.samples / frame.sample_rate == pytest.approx(AUDIO_PTIME)

    assert track.frames_sent == 5
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_pacing_does_not_drift() -> None:
    track, clock = make_track()

    for _ in range(50):
        await track.recv()

    # 49 sleeps: the first is a full interval (the frame before it was emitted
    # on time), every later one is shortened by the 4 ms wakeup overhead
    # instead of adding it to the next deadline.
    assert len(clock.sleeps) == 49
    assert clock.sleeps[0] == pytest.approx(AUDIO_PTIME)
    for delay in clock.sleeps[1:]:
        assert delay == pytest.approx(AUDIO_PTIME - clock.overhead)
    # The schedule is still exactly on the 20 ms grid it started on.
    assert track.pacer.deadline == pytest.approx(500.0 + 50 * AUDIO_PTIME)
    assert track.pacer.resyncs == 0
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_recovers_from_a_stalled_loop() -> None:
    track, clock = make_track()
    await track.recv()
    clock.now += 3.0  # the loop was blocked
    await track.recv()
    assert track.pacer.resyncs == 1
    assert clock.sleeps == []  # no sleep, and no backlog burst either
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_samples_are_near_silence() -> None:
    track, _clock = make_track(amplitude=1)
    frame = await track.recv()
    data = bytes(frame.planes[0])
    assert len(data) == 960 * 2 * 2
    samples = [
        int.from_bytes(data[i : i + 2], "little", signed=True)
        for i in range(0, len(data), 2)
    ]
    assert set(samples) == {1, -1}
    assert sum(samples) == 0  # no DC offset
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_amplitude_zero_is_exact_silence() -> None:
    track, _clock = make_track(amplitude=0)
    frame = await track.recv()
    assert bytes(frame.planes[0]) == bytes(960 * 2 * 2)
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_frames_encode_as_opus() -> None:
    """Proves the frame shape is what aiortc's sender will hand to libopus."""
    from aiortc.codecs.opus import OpusEncoder

    track, _clock = make_track()
    encoder = OpusEncoder()
    sizes: list[int] = []
    for _ in range(5):
        packets, _timestamp = encoder.encode(await track.recv())
        sizes += [len(p) for p in packets]

    assert len(sizes) == 5, "one Opus packet per 20 ms frame"
    assert all(size > 0 for size in sizes)
    track.stop()


@pytest.mark.asyncio
async def test_silence_track_stops() -> None:
    track, _clock = make_track()
    await track.recv()
    track.stop()
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_muted_leg_sends_exact_silence() -> None:
    leg = AudioLeg(muted=True, ice_servers=[])
    try:
        assert leg.muted is True
        assert leg.track.amplitude == 0
    finally:
        await leg.close()

    leg = AudioLeg(muted=False, amplitude=2, ice_servers=[])
    try:
        assert leg.track.amplitude == 2
    finally:
        await leg.close()


# --------------------------------------------------------------------------- #
# Chime TURN relay (regression: live 1006 drop while waiting for presence)
# --------------------------------------------------------------------------- #


def _turn_creds(uris=("turn:10.0.0.1:3478?transport=udp",), user="u", pw="p"):
    from huddle_drawings.proto import signaling_protocol_pb2 as sp

    return sp.SdkTurnCredentials(username=user, password=pw, ttl=300, uris=list(uris))


def test_ice_servers_from_turn_credentials() -> None:
    from huddle_drawings.audio import ice_servers_from_turn_credentials

    servers = ice_servers_from_turn_credentials(
        _turn_creds(("turn:a:3478?transport=udp", "turns:a:443?transport=tcp"))
    )
    assert len(servers) == 1
    assert servers[0].username == "u"
    assert servers[0].credential == "p"
    assert list(servers[0].urls) == [
        "turn:a:3478?transport=udp",
        "turns:a:443?transport=tcp",
    ]


def test_ice_servers_from_missing_or_empty_credentials() -> None:
    from huddle_drawings.proto import signaling_protocol_pb2 as sp

    from huddle_drawings.audio import ice_servers_from_turn_credentials

    assert ice_servers_from_turn_credentials(None) == []
    assert ice_servers_from_turn_credentials(sp.SdkTurnCredentials()) == []


@pytest.mark.asyncio
async def test_turn_servers_force_relay_only_transport_policy() -> None:
    """Chime media only traverses its own TURN relay.

    ``CreatePeerConnectionTask`` in amazon-chime-sdk-js sets
    ``iceTransportPolicy: 'relay'`` alongside the JOIN_ACK TURN credentials.
    Offering host/srflx candidates instead gets no media through and Chime
    closes the signaling socket with 1006 a couple of seconds in -- which
    looks exactly like the SRTP failure and is not. Regression test for that
    live failure.
    """
    from huddle_drawings.audio import AudioLeg, ice_servers_from_turn_credentials

    leg = AudioLeg(
        ice_servers=ice_servers_from_turn_credentials(
            _turn_creds(("turn:1.2.3.4:3478?transport=udp",))
        )
    )
    try:
        assert leg.relay_only is True
        # Assert on the ICE layer rather than the stored config: this proves
        # the policy and the TURN server actually reached aioice, which is
        # what decides whether any media flows.
        connection = leg.pc.getTransceivers()[0].receiver.transport.transport._connection
        # aioice keeps the policy private; the turn_* attributes are public.
        assert connection._transport_policy.name == "RELAY"
        assert connection.turn_server == ("1.2.3.4", 3478)
        assert connection.turn_username == "u"
        assert connection.turn_password == "p"
    finally:
        await leg.close()


@pytest.mark.asyncio
async def test_stun_only_or_empty_servers_do_not_force_relay() -> None:
    from aiortc import RTCIceServer

    from huddle_drawings.audio import AudioLeg

    for servers in ([], [RTCIceServer("stun:stun.example:19302")]):
        leg = AudioLeg(ice_servers=servers)
        try:
            assert leg.relay_only is False, servers
        finally:
            await leg.close()


@pytest.mark.asyncio
async def test_relay_only_can_be_overridden() -> None:
    from huddle_drawings.audio import AudioLeg, ice_servers_from_turn_credentials

    leg = AudioLeg(
        ice_servers=ice_servers_from_turn_credentials(_turn_creds()),
        relay_only=False,
    )
    try:
        assert leg.relay_only is False
    finally:
        await leg.close()


# --------------------------------------------------------------------------- #
# Live-found aiortc <-> Chime DTLS incompatibilities
#
# Both were diagnosed against a real huddle on 2026-11-08 and are the reason a
# media leg previously never connected. Neither is mentioned in the vendored
# docs.
# --------------------------------------------------------------------------- #


def test_dtls_cipher_list_includes_ecdhe_rsa() -> None:
    """Chime's media server has an RSA certificate.

    We are the DTLS client, so the negotiated suite must suit *Chime's*
    certificate. aiortc offers only ECDHE-ECDSA suites, so there is no overlap
    and Chime answers with a fatal handshake_failure (alert 40).
    """
    from huddle_drawings.audio import CHIME_DTLS_CIPHERS

    suites = CHIME_DTLS_CIPHERS.decode().split(":")
    assert any(s.startswith("ECDHE-RSA-") for s in suites), suites
    # ECDSA must stay too: a future Chime cert may be ECDSA, and other peers are.
    assert any(s.startswith("ECDHE-ECDSA-") for s in suites), suites


def test_force_chime_dtls_ciphers_installs_an_idempotent_wrapper() -> None:
    from aiortc.rtcdtlstransport import (
        SRTP_AES128_CM_SHA1_80,
        RTCCertificate,
    )

    from huddle_drawings.audio import CHIME_DTLS_CIPHERS, force_chime_dtls_ciphers

    force_chime_dtls_ciphers()
    first = RTCCertificate._create_ssl_context
    force_chime_dtls_ciphers()
    assert RTCCertificate._create_ssl_context is first, "must not re-wrap"
    assert getattr(first, "_huddle_drawings_patched", False)

    # A real context must still build through the wrapper, and OpenSSL must
    # accept our cipher list -- an invalid suite name would raise here, which
    # is the failure mode worth catching.
    cert = RTCCertificate.generateCertificate()
    ctx = cert._create_ssl_context([SRTP_AES128_CM_SHA1_80])
    assert ctx is not None
    ctx.set_cipher_list(CHIME_DTLS_CIPHERS)  # accepted by OpenSSL


def test_dtls_cipher_list_is_accepted_by_openssl() -> None:
    """Guard against a typo silently disabling every suite."""
    from OpenSSL import SSL

    from huddle_drawings.audio import CHIME_DTLS_CIPHERS

    ctx = SSL.Context(SSL.DTLS_METHOD)
    ctx.set_cipher_list(CHIME_DTLS_CIPHERS)


def test_keep_single_fingerprint_drops_the_others() -> None:
    """Chime rejects aiortc's three-fingerprint offer with a fatal unknown_ca."""
    from huddle_drawings.audio import keep_single_fingerprint

    sdp = "\r\n".join(
        [
            "v=0",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111",
            "a=fingerprint:sha-256 AA:BB",
            "a=fingerprint:sha-384 CC:DD",
            "a=fingerprint:sha-512 EE:FF",
            "a=setup:actpass",
            "m=video 9 UDP/TLS/RTP/SAVPF 99",
            "a=fingerprint:sha-256 AA:BB",
            "a=fingerprint:sha-384 CC:DD",
        ]
    ) + "\r\n"
    out = keep_single_fingerprint(sdp)
    assert out.count("a=fingerprint:") == 2
    assert "sha-384" not in out
    assert "sha-512" not in out
    assert "a=fingerprint:sha-256 AA:BB" in out
    assert out.endswith("\r\n")
    assert keep_single_fingerprint(out) == out  # idempotent


def test_keep_single_fingerprint_leaves_sdp_alone_when_algorithm_absent() -> None:
    """Never strip every fingerprint: an SDP without one cannot do DTLS."""
    from huddle_drawings.audio import keep_single_fingerprint

    sdp = "v=0\r\nm=audio 9 x\r\na=fingerprint:sha-512 EE:FF\r\n"
    assert keep_single_fingerprint(sdp) == sdp


@pytest.mark.asyncio
async def test_offer_carries_exactly_one_fingerprint_algorithm(leg: AudioLeg) -> None:
    sdp = await leg.create_offer()
    algorithms = {
        line.split()[0].removeprefix("a=fingerprint:")
        for line in sdp.splitlines()
        if line.startswith("a=fingerprint:")
    }
    assert algorithms == {"sha-256"}, algorithms


def test_propagate_bundle_candidates_copies_onto_every_bundled_section() -> None:
    """Works around aiortc losing candidates under max-bundle.

    aiortc keys remote candidates by ICE transport
    (``iceCandidates[iceTransport] = media``); under max-bundle every section
    shares one transport, so the last section wins. Chime puts candidates only
    on the primary section, so the survivor is empty and ICE never starts.
    """
    from huddle_drawings.audio import propagate_bundle_candidates

    answer = "\r\n".join(
        [
            "v=0",
            "a=group:BUNDLE 0 1",
            "m=audio 1 x",
            "a=candidate:1 1 udp 1 10.0.0.1 1 typ relay",
            "a=mid:0",
            "m=video 9 x",
            "a=mid:1",
        ]
    ) + "\r\n"
    out = propagate_bundle_candidates(answer)
    audio_part, video_part = out.split("m=video")
    assert "a=candidate:" in audio_part
    assert "a=candidate:" in video_part
    assert propagate_bundle_candidates(out) == out  # idempotent


def test_propagate_bundle_candidates_noop_without_bundle_or_candidates() -> None:
    from huddle_drawings.audio import propagate_bundle_candidates

    no_bundle = "v=0\r\nm=audio 1 x\r\na=candidate:1 1 udp 1 1.2.3.4 1 typ relay\r\n"
    assert propagate_bundle_candidates(no_bundle) == no_bundle

    no_candidates = "v=0\r\na=group:BUNDLE 0 1\r\nm=audio 1 x\r\nm=video 9 x\r\n"
    assert propagate_bundle_candidates(no_candidates) == no_candidates

    assert propagate_bundle_candidates("v=0\r\n") == "v=0\r\n"


@pytest.mark.asyncio
async def test_relay_only_offer_without_relay_candidates_fails_fast() -> None:
    """A failed TURN allocation must be reported at the offer, not 15s later."""
    from aiortc import RTCIceServer

    from huddle_drawings.audio import AudioLeg, MediaConnectionError

    # A TURN server that cannot be reached yields no relay candidates under
    # relay-only ICE (aioice gathers no host/srflx candidates at all).
    leg = AudioLeg(
        ice_servers=[
            RTCIceServer(
                urls=["turn:127.0.0.1:1?transport=udp"], username="u", credential="p"
            )
        ],
    )
    try:
        assert leg.relay_only is True
        with pytest.raises(MediaConnectionError, match="no relay ICE candidates"):
            await leg.create_offer()
    finally:
        await leg.close()
