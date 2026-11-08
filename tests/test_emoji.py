"""Tests for :mod:`huddle_drawings.emoji`.

No network and no Slack identity: downloads go through ``httpx.MockTransport``
and the Slack side is a four-line fake exposing just ``resolve_emoji``, which
is all :func:`~huddle_drawings.emoji.emoji_image` asks for.

Font-backed tests are skipped when the box has no colour emoji font, so the
suite still passes on a machine without one -- that degradation is itself
tested (:func:`test_font_missing_is_a_clear_error`), with
:func:`~huddle_drawings.emoji.find_emoji_font` monkeypatched to ``None``.
"""

from __future__ import annotations

import io
import logging
import unicodedata
from collections.abc import Callable

import httpx
import numpy as np
import pytest
from PIL import Image

from huddle_drawings import emoji as em
from huddle_drawings import image_trace as it
from huddle_drawings.types import HuddleError

TADA = "\N{PARTY POPPER}"
THUMBSUP = "\N{THUMBS UP SIGN}"
FAMILY = "\N{WOMAN}‍\N{WOMAN}‍\N{GIRL}"
FLAG_JP = (
    "\N{REGIONAL INDICATOR SYMBOL LETTER J}"
    "\N{REGIONAL INDICATOR SYMBOL LETTER P}"
)
HEART = "\N{HEAVY BLACK HEART}\N{VARIATION SELECTOR-16}"
KEYCAP_ONE = "1\N{VARIATION SELECTOR-16}\N{COMBINING ENCLOSING KEYCAP}"
SHRUG_WOMAN = (
    "\N{SHRUG}‍\N{FEMALE SIGN}\N{VARIATION SELECTOR-16}"
)

FONT = em.find_emoji_font()
needs_font = pytest.mark.skipif(
    FONT is None, reason="no colour emoji font installed on this box"
)

SQUIRREL_URL = "https://emoji.slack-edge.com/T000/squirrel/abc123.png"
CUSTOM_URL = "https://emoji.slack-edge.com/T000/my-custom-thing/def456.png"


# --------------------------------------------------------------------------
# fixtures: every image is built here, in code
# --------------------------------------------------------------------------


class FakeSlack:
    """The ``resolve_emoji`` half of ``SlackClient``, with a dict for a map.

    Raises the same :class:`HuddleError` the real client raises for a name the
    workspace does not list -- which is the *normal* case for part of the
    standard set (``:tada:`` really is absent from a real 65k-entry map).
    """

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = dict(table or {})
        self.calls: list[str] = []

    async def resolve_emoji(self, name: str, *, max_hops: int = 8) -> str:
        key = name.strip().strip(":").strip().lower()
        self.calls.append(key)
        url = self.table.get(key)
        if url is None:
            raise HuddleError(f"this workspace has no emoji called :{key}:")
        return url


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def animated_gif_bytes(
    colors: tuple[tuple[int, int, int], ...] = ((255, 0, 0), (0, 0, 255)),
    size: tuple[int, int] = (32, 32),
) -> bytes:
    frames = [Image.new("RGB", size, c) for c in colors]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    return buf.getvalue()


def padded_square(
    *,
    box: int = 64,
    glyph: int = 16,
    offset: int = 8,
    color: tuple[int, int, int] = (0, 64, 255),
) -> Image.Image:
    """A small opaque square inside a transparent box -- a Slack PNG, basically."""
    img = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    img.paste(
        Image.new("RGBA", (glyph, glyph), (*color, 255)), (offset, offset)
    )
    return img


def holed_square(
    *, box: int = 64, hole: int = 16, color: tuple[int, int, int] = (255, 0, 0)
) -> Image.Image:
    """Opaque everywhere except a transparent corner block.

    The alpha bbox is the whole image, so trimming is a no-op and the only
    thing that can clear the corner is compositing.
    """
    img = Image.new("RGBA", (box, box), (*color, 255))
    img.paste(Image.new("RGBA", (hole, hole), (0, 0, 0, 0)), (0, 0))
    return img


def make_client(
    table: dict[str, tuple[bytes, str]],
    *,
    record: list[str] | None = None,
    status: int = 200,
) -> httpx.AsyncClient:
    """An ``AsyncClient`` that serves ``table`` and 404s everything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if record is not None:
            record.append(url)
        entry = table.get(url)
        if entry is None:
            return httpx.Response(404, text="not found")
        content, content_type = entry
        return httpx.Response(
            status, content=content, headers={"content-type": content_type}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def handler_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def pixels(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.int16)


def unique_colors(img: Image.Image) -> int:
    flat = pixels(img.convert("RGB")).reshape(-1, 3)
    return len(np.unique(flat, axis=0))


# --------------------------------------------------------------------------
# classifying a spec
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        TADA,
        THUMBSUP,
        FAMILY,
        FLAG_JP,
        HEART,
        KEYCAP_ONE,
        SHRUG_WOMAN,
        "\N{THUMBS UP SIGN}\N{EMOJI MODIFIER FITZPATRICK TYPE-4}",
        "\N{WHITE HEAVY CHECK MARK}",
        "\N{HIGH VOLTAGE SIGN}",
        "\N{WHITE LARGE SQUARE}",
        f"  {TADA}  ",
    ],
)
def test_looks_like_unicode_emoji_accepts_characters_and_sequences(spec: str) -> None:
    assert em.looks_like_unicode_emoji(spec) is True


@pytest.mark.parametrize(
    "spec",
    [
        "tada",
        "hello",
        "",
        "   ",
        ":tada:",
        ":my-custom-thing:",
        "+1",
        "-1",
        "100",
        "a",
        "A",
        "1",
        "→",  # RIGHTWARDS ARROW: a text arrow, not an emoji one
        "⌣",  # SMILE, the maths symbol unicodedata finds for ":smile:"
        "⌂",  # HOUSE, likewise for ":house:"
        "‍",  # a lone ZWJ is not a pictograph
        f"{TADA} and some words",
        TADA * 40,
    ],
)
def test_looks_like_unicode_emoji_rejects_names_and_text(spec: str) -> None:
    assert em.looks_like_unicode_emoji(spec) is False


def test_unicodedata_near_misses_are_not_pictographs() -> None:
    """The reason step 3 validates its own hit before rendering it.

    ``unicodedata.lookup("SMILE")`` happily returns a maths symbol, and
    ``"HOUSE"`` a technical one. No emoji font draws either, so accepting them
    would mean a blank drawing instead of a fallback.
    """
    for name in ("SMILE", "HOUSE"):
        assert not em.looks_like_unicode_emoji(unicodedata.lookup(name))


# --------------------------------------------------------------------------
# the local font
# --------------------------------------------------------------------------


def test_font_candidates_start_with_noto() -> None:
    assert em.EMOJI_FONT_CANDIDATES
    assert "NotoColorEmoji" in em.EMOJI_FONT_CANDIDATES[0]


def test_find_emoji_font_returns_an_existing_file_or_none() -> None:
    found = em.find_emoji_font()
    assert found is None or found.is_file()


def test_find_emoji_font_is_none_when_no_candidate_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        em, "EMOJI_FONT_CANDIDATES", (str(tmp_path / "nope.ttf"),)
    )
    assert em.find_emoji_font() is None


@needs_font
@pytest.mark.parametrize("size", [32, 64, 128, 512])
def test_render_unicode_emoji_honours_size_and_keeps_alpha(size: int) -> None:
    img = em.render_unicode_emoji(TADA, size=size)
    assert img.mode == "RGBA"
    assert max(img.size) == size
    # Cropped to the glyph: no fully transparent border row/column left.
    assert img.getchannel("A").getbbox() == (0, 0, img.width, img.height)


@needs_font
def test_render_unicode_emoji_is_not_blank_and_has_colour() -> None:
    img = em.render_unicode_emoji(TADA, size=128)
    assert unique_colors(img) > 8


@needs_font
def test_render_unicode_emoji_rejects_a_missing_glyph() -> None:
    """A character the emoji font does not cover renders as nothing at all."""
    with pytest.raises(HuddleError, match="no glyph"):
        em.render_unicode_emoji("⌣", size=64)


def test_render_unicode_emoji_rejects_empty_text() -> None:
    with pytest.raises(HuddleError, match="empty"):
        em.render_unicode_emoji("", size=64)


def test_render_unicode_emoji_rejects_a_bad_size() -> None:
    with pytest.raises(ValueError, match="size must be"):
        em.render_unicode_emoji(TADA, size=0)


def test_render_unicode_emoji_rejects_a_missing_font_path(tmp_path) -> None:
    with pytest.raises(HuddleError, match="does not exist"):
        em.render_unicode_emoji(TADA, size=64, font_path=tmp_path / "nope.ttf")


def test_render_unicode_emoji_rejects_a_file_that_is_not_a_font(tmp_path) -> None:
    fake = tmp_path / "fake.ttf"
    fake.write_bytes(b"this is not a font")
    with pytest.raises(HuddleError, match="could not be opened as a font"):
        em.render_unicode_emoji(TADA, size=64, font_path=fake)


# --------------------------------------------------------------------------
# step 1: a literal emoji character
# --------------------------------------------------------------------------


@needs_font
async def test_literal_emoji_renders_a_multicolour_image_at_the_requested_size() -> None:
    img = await em.emoji_image(TADA, size=256)
    assert img.mode == "RGB"
    assert max(img.size) == 256
    assert unique_colors(img) > 16  # not blank, and genuinely in colour
    # and not just a flat background
    assert pixels(img).std() > 5


@needs_font
@pytest.mark.parametrize("spec", [TADA, THUMBSUP, FAMILY, FLAG_JP, HEART])
async def test_sequences_render_as_one_image(spec: str) -> None:
    render = await em.resolve_emoji_image(spec, size=128)
    assert render.kind == "unicode"
    assert render.character == spec
    assert max(render.image.size) == 128
    assert unique_colors(render.image) > 2


@needs_font
async def test_size_is_clamped_to_max_size() -> None:
    img = await em.emoji_image(TADA, size=em.MAX_SIZE * 4)
    assert max(img.size) == em.MAX_SIZE


@needs_font
async def test_background_is_honoured() -> None:
    white = await em.emoji_image(TADA, size=96)
    black = await em.emoji_image(TADA, size=96, background=(0, 0, 0))
    named = await em.emoji_image(TADA, size=96, background="black")
    assert pixels(black).mean() < pixels(white).mean()
    assert np.array_equal(pixels(black), pixels(named))


async def test_a_bad_background_is_rejected() -> None:
    with pytest.raises(ValueError, match="background must be"):
        await em.emoji_image(TADA, background=(255, 255))  # type: ignore[arg-type]


@needs_font
async def test_provenance_is_logged_and_attached(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="huddle_drawings.emoji")
    img = await em.emoji_image(TADA, size=64)
    assert img.info["emoji_spec"] == TADA
    assert img.info["emoji_source"] == "unicode"
    assert "literal character" in img.info["emoji_detail"]
    assert any("via unicode" in r.getMessage() for r in caplog.records)


@needs_font
async def test_the_result_traces_in_colour() -> None:
    """The point of the module: hand the image straight to the tracer."""
    img = await em.emoji_image(TADA, size=512)
    layers = it.trace_image_colors(img, max_segments=600)
    assert layers, "the party popper should decompose into at least one layer"
    assert sum(len(layer.segments) for layer in layers) <= 600


async def test_empty_spec_is_rejected() -> None:
    for spec in ("", "   ", ":", "::"):
        with pytest.raises(HuddleError, match="empty emoji spec|not an emoji"):
            await em.emoji_image(spec)


# --------------------------------------------------------------------------
# step 2: a Slack emoji name
# --------------------------------------------------------------------------


async def test_slack_name_resolves_and_downloads() -> None:
    seen: list[str] = []
    slack = FakeSlack({"squirrel": SQUIRREL_URL})
    client = make_client(
        {SQUIRREL_URL: (png_bytes(padded_square()), "image/png")}, record=seen
    )
    async with client:
        render = await em.resolve_emoji_image(
            ":squirrel:", slack=slack, client=client, size=64
        )
    assert render.kind == "slack"
    assert render.url == SQUIRREL_URL
    assert seen == [SQUIRREL_URL]
    assert slack.calls == ["squirrel"]
    assert render.image.mode == "RGB"
    assert render.image.info["emoji_source"] == "slack"


@pytest.mark.parametrize("spec", ["squirrel", ":squirrel:", " :SQUIRREL: ", "Squirrel"])
async def test_colons_and_case_are_stripped_from_names(spec: str) -> None:
    slack = FakeSlack({"squirrel": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        render = await em.resolve_emoji_image(
            spec, slack=slack, client=client, size=32
        )
    assert render.kind == "slack"


async def test_custom_emoji_names_with_dashes_work() -> None:
    slack = FakeSlack({"my-custom-thing": CUSTOM_URL})
    client = make_client({CUSTOM_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        render = await em.resolve_emoji_image(
            ":my-custom-thing:", slack=slack, client=client, size=32
        )
    assert render.kind == "slack"
    assert render.url == CUSTOM_URL


async def test_slack_wins_over_unicodedata_when_the_workspace_has_the_name() -> None:
    """``:fire:`` is both in ``emoji.list`` and a Unicode character name."""
    url = "https://emoji.slack-edge.com/T000/fire/aaa.png"
    slack = FakeSlack({"fire": url})
    client = make_client({url: (png_bytes(padded_square()), "image/png")})
    async with client:
        render = await em.resolve_emoji_image(
            "fire", slack=slack, client=client, size=32
        )
    assert render.kind == "slack"
    assert render.attempts == ()


async def test_an_alpha_png_is_composited_onto_white() -> None:
    slack = FakeSlack({"holed": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (png_bytes(holed_square()), "image/png")})
    async with client:
        img = await em.emoji_image("holed", slack=slack, client=client, size=64)
    assert img.mode == "RGB"
    assert img.getbands() == ("R", "G", "B")
    data = pixels(img)
    # the transparent corner is white now, the opaque body is still red
    assert tuple(data[2, 2]) == (255, 255, 255)
    assert tuple(data[-3, -3]) == (255, 0, 0)


async def test_transparent_margins_are_trimmed_before_scaling() -> None:
    """A 16px glyph in a 64px box must fill the result, not a quarter of it."""
    slack = FakeSlack({"padded": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        img = await em.emoji_image("padded", slack=slack, client=client, size=32)
    assert img.size == (32, 32)
    data = pixels(img)
    # every pixel is the glyph colour: none of the padding survived
    assert not np.any(np.all(data == 255, axis=-1)), "white padding was kept"
    assert tuple(data[0, 0]) == (0, 64, 255)
    assert tuple(data[-1, -1]) == (0, 64, 255)


async def test_a_small_png_is_upscaled_for_the_tracer() -> None:
    slack = FakeSlack({"tiny": SQUIRREL_URL})
    client = make_client(
        {SQUIRREL_URL: (png_bytes(padded_square(box=64, glyph=48, offset=8)), "image/png")}
    )
    async with client:
        img = await em.emoji_image("tiny", slack=slack, client=client, size=512)
    assert max(img.size) == 512


async def test_an_animated_gif_yields_one_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="huddle_drawings.emoji")
    slack = FakeSlack({"spin": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (animated_gif_bytes(), "image/gif")})
    async with client:
        render = await em.resolve_emoji_image(
            "spin", slack=slack, client=client, size=16
        )
    assert render.animated is True
    data = pixels(render.image)
    assert data[..., 0].mean() > 200, "frame 0 is the red one"
    assert data[..., 2].mean() < 60, "frame 1 (blue) must not be blended in"
    assert any("first frame" in r.getMessage() for r in caplog.records)


async def test_a_still_gif_is_not_reported_as_animated() -> None:
    slack = FakeSlack({"still": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (animated_gif_bytes(((0, 200, 0),)), "image/gif")})
    async with client:
        render = await em.resolve_emoji_image(
            "still", slack=slack, client=client, size=16
        )
    assert render.animated is False


async def test_a_jpeg_without_alpha_works() -> None:
    slack = FakeSlack({"photo": SQUIRREL_URL})
    body = jpeg_bytes(Image.new("RGB", (48, 48), (12, 200, 90)))
    client = make_client({SQUIRREL_URL: (body, "image/jpeg")})
    async with client:
        img = await em.emoji_image("photo", slack=slack, client=client, size=64)
    assert img.mode == "RGB"
    assert max(img.size) == 64


# --------------------------------------------------------------------------
# downloading: failures
# --------------------------------------------------------------------------


async def test_a_non_image_body_raises_with_the_url() -> None:
    client = make_client({SQUIRREL_URL: (b"<html>signin</html>", "text/html")})
    async with client:
        with pytest.raises(HuddleError) as excinfo:
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)
    assert SQUIRREL_URL in str(excinfo.value)
    assert "decodable image" in str(excinfo.value)


async def test_a_non_image_body_raises_through_emoji_image() -> None:
    slack = FakeSlack({"broken": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (b"nope", "text/plain")})
    async with client:
        with pytest.raises(HuddleError, match="decodable image"):
            await em.emoji_image("broken", slack=slack, client=client)


async def test_an_http_error_raises_with_the_status() -> None:
    client = make_client({})
    async with client:
        with pytest.raises(HuddleError, match="HTTP 404"):
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)


async def test_a_transport_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = handler_client(handler)
    async with client:
        with pytest.raises(HuddleError, match="failed"):
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)


async def test_an_empty_body_raises() -> None:
    client = make_client({SQUIRREL_URL: (b"", "image/png")})
    async with client:
        with pytest.raises(HuddleError, match="empty body"):
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)


async def test_an_oversized_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em, "MAX_DOWNLOAD_BYTES", 32)
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        with pytest.raises(HuddleError, match="over the"):
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)


async def test_an_absurdly_large_image_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(em, "MAX_SOURCE_DIMENSION", 16)
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square(box=64)), "image/png")})
    async with client:
        with pytest.raises(HuddleError, match="not a sane size"):
            await em.fetch_emoji_image(SQUIRREL_URL, client=client)


@pytest.mark.parametrize("url", ["", "file:///etc/passwd", "emoji.png", "ftp://x/y.png"])
async def test_a_non_http_url_is_refused(url: str) -> None:
    with pytest.raises(HuddleError, match="not an http"):
        await em.fetch_emoji_image(url)


async def test_a_fully_transparent_download_raises() -> None:
    slack = FakeSlack({"ghost": SQUIRREL_URL})
    blank = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    client = make_client({SQUIRREL_URL: (png_bytes(blank), "image/png")})
    async with client:
        with pytest.raises(HuddleError, match="fully transparent"):
            await em.emoji_image("ghost", slack=slack, client=client)


async def test_fetch_returns_rgba_and_leaves_an_injected_client_open() -> None:
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        img = await em.fetch_emoji_image(SQUIRREL_URL, client=client)
        assert img.mode == "RGBA"
        assert img.info["emoji_url"] == SQUIRREL_URL
        # still usable: fetch_emoji_image must not close a borrowed client
        again = await em.fetch_emoji_image(SQUIRREL_URL, client=client)
        assert again.size == (64, 64)


# --------------------------------------------------------------------------
# steps 3 and 4: the fallbacks for names Slack does not list
# --------------------------------------------------------------------------


@needs_font
@pytest.mark.parametrize(
    ("spec", "char"),
    [
        ("party_popper", TADA),
        (":party_popper:", TADA),
        ("party-popper", TADA),
        ("rocket", "\N{ROCKET}"),
        ("snowflake", "\N{SNOWFLAKE}"),
        ("face_with_tears_of_joy", "\N{FACE WITH TEARS OF JOY}"),
    ],
)
async def test_a_name_missing_from_the_workspace_falls_back_to_unicodedata(
    spec: str, char: str
) -> None:
    slack = FakeSlack()  # empty workspace: nothing is in emoji.list
    render = await em.resolve_emoji_image(spec, slack=slack, size=96)
    assert render.kind == "unicodedata"
    assert render.character == char
    assert unique_colors(render.image) > 2
    assert any("emoji.list" in attempt for attempt in render.attempts)


@needs_font
@pytest.mark.parametrize(
    ("spec", "char"),
    [
        ("tada", TADA),
        (":tada:", TADA),
        ("+1", THUMBSUP),
        ("thumbsup", THUMBSUP),
        ("-1", "\N{THUMBS DOWN SIGN}"),
        ("100", "\N{HUNDRED POINTS SYMBOL}"),
        ("joy", "\N{FACE WITH TEARS OF JOY}"),
        ("smile", "\N{SMILING FACE WITH OPEN MOUTH AND SMILING EYES}"),
    ],
)
async def test_a_name_unicodedata_cannot_find_falls_back_to_the_curated_table(
    spec: str, char: str
) -> None:
    slack = FakeSlack()
    render = await em.resolve_emoji_image(spec, slack=slack, size=96)
    assert render.kind == "alias"
    assert render.character == char
    assert unique_colors(render.image) > 2


@needs_font
async def test_smile_does_not_become_the_maths_symbol() -> None:
    """Step 3 must not win here: ``unicodedata.lookup("SMILE")`` is ``⌣``."""
    render = await em.resolve_emoji_image("smile", slack=FakeSlack(), size=64)
    assert render.kind == "alias"
    assert render.character != "⌣"


@needs_font
async def test_the_fallbacks_work_with_no_slack_client_at_all() -> None:
    render = await em.resolve_emoji_image("tada", size=64)
    assert render.kind == "alias"
    assert any("no Slack client" in attempt for attempt in render.attempts)


@needs_font
@pytest.mark.parametrize("name", sorted(em.SHORTCODE_ALIASES))
def test_every_curated_alias_renders(name: str) -> None:
    char = em.SHORTCODE_ALIASES[name]
    assert em.looks_like_unicode_emoji(char), f":{name}: -> {char!r}"
    img = em.render_unicode_emoji(char, size=64)
    assert img.getchannel("A").getbbox() is not None


def test_the_curated_table_is_short_and_normalised() -> None:
    assert len(em.SHORTCODE_ALIASES) < 80, "this table is deliberately not exhaustive"
    for name in em.SHORTCODE_ALIASES:
        assert name == name.strip().lower()
        assert ":" not in name


# --------------------------------------------------------------------------
# step 5: giving up, and the no-font box
# --------------------------------------------------------------------------


async def test_an_unknown_name_raises_saying_what_was_tried() -> None:
    slack = FakeSlack({"squirrel": SQUIRREL_URL})
    with pytest.raises(HuddleError) as excinfo:
        await em.emoji_image("definitely-not-an-emoji", slack=slack)
    message = str(excinfo.value)
    assert "definitely-not-an-emoji" in message
    assert "emoji.list" in message
    assert "unicodedata" in message
    assert "curated" in message
    assert "literal emoji character" in message


async def test_font_missing_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No emoji font must degrade to a readable HuddleError, not a traceback."""
    monkeypatch.setattr(em, "find_emoji_font", lambda: None)
    with pytest.raises(HuddleError) as excinfo:
        await em.emoji_image(TADA)
    message = str(excinfo.value)
    assert "no colour emoji font" in message
    assert "NotoColorEmoji" in message
    assert "font_path" in message

    with pytest.raises(HuddleError, match="no colour emoji font"):
        em.render_unicode_emoji(TADA, size=64)


async def test_font_missing_still_reports_what_was_tried_for_a_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(em, "find_emoji_font", lambda: None)
    with pytest.raises(HuddleError) as excinfo:
        await em.emoji_image("tada", slack=FakeSlack())
    message = str(excinfo.value)
    assert "curated alias" in message
    assert "no colour emoji font" in message


async def test_font_missing_does_not_break_the_slack_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A downloaded emoji needs no font at all."""
    monkeypatch.setattr(em, "find_emoji_font", lambda: None)
    slack = FakeSlack({"squirrel": SQUIRREL_URL})
    client = make_client({SQUIRREL_URL: (png_bytes(padded_square()), "image/png")})
    async with client:
        render = await em.resolve_emoji_image(
            ":squirrel:", slack=slack, client=client, size=64
        )
    assert render.kind == "slack"
    assert render.image.mode == "RGB"
