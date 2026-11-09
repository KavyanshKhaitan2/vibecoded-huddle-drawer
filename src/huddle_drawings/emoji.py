"""Turn an emoji into a PIL image the tracer can draw in colour.

Three kinds of input all end up as one opaque ``RGB`` :class:`PIL.Image.Image`
that :func:`huddle_drawings.image_trace.trace_image` /
:func:`~huddle_drawings.image_trace.trace_image_colors` accept directly (both
take ``str | Path | Image.Image``):

* a **literal emoji character** -- ``"\N{PARTY POPPER}"``, a ZWJ family, a flag
  -- rendered from a local colour emoji font;
* a **Slack emoji name** -- ``":squirrel:"``, ``":my-custom-thing:"`` --
  resolved through :meth:`huddle_drawings.slack.SlackClient.resolve_emoji`
  (which follows ``alias:`` chains) and downloaded from
  ``emoji.slack-edge.com``;
* a **standard shortcode the workspace does not list** -- Slack renders a chunk
  of the standard set from its own sprite rather than from ``emoji.list``, so
  ``:tada:`` is genuinely *absent* from a real workspace's 65k-entry map. Those
  fall back to :func:`unicodedata.lookup` on the name, then to a small curated
  alias table (:data:`SHORTCODE_ALIASES`).

Nothing here traces. :func:`emoji_image` produces pixels;
:mod:`huddle_drawings.image_trace` turns pixels into segments.

**Why the image comes back opaque.** Downloaded emoji are nearly always PNGs
with a transparent background, and font glyphs are rendered onto nothing at
all. The tracer composites alpha onto white itself, but it does that *after*
deciding what to trace, so a transparent image handed to the colour tracer
clusters its background as one big dark region and the drawing comes out as a
solid block. So this module composites onto ``background`` (white by default)
and returns ``RGB``. It also **trims fully transparent margins first**, because
a 64 px Slack PNG is typically a 50 px glyph in a box of padding and the
tracer would otherwise scale the padding, not the emoji.

**Why the font is rendered at a fixed size and upscaled.**
``NotoColorEmoji.ttf`` -- the only emoji font on most Linux boxes -- is a CBDT
*bitmap* font: Pillow will only open it at its one embedded strike (109 px on
the version seen here) and raises ``OSError: invalid pixel size`` for anything
else. So :func:`render_unicode_emoji` opens the requested size first (which
works for a scalable COLR/SVG font), falls back to the known fixed strikes, and
LANCZOS-upscales the result to the requested size. At 109 px a party popper
still has ~150 distinct colours, so the colour information survives.

**Known rough edges** (all in what the *emoji* look like, not in this module):

* ``NotoColorEmoji`` draws multi-person **ZWJ sequences** (``👩‍👩‍👧``) as a
  grey silhouette, not in colour, so they come back as one near-greyscale layer
  from :func:`~huddle_drawings.image_trace.trace_image_colors`. Single-person
  emoji are in full colour.
* Emoji that are **mostly white** (flags like ``🇯🇵``, ``⬜``, ``🥚``) have
  almost nothing drawable: the colour tracer cannot draw a near-white region at
  all (it documents that), so a Japanese flag traces as a red disc with no
  border.
* Many emoji are **one hue** (``👍`` is all amber), so a colour trace produces a
  single layer -- i.e. one attendee's worth of drawing, not several.
* The step-3 :func:`unicodedata.lookup` fallback resolves the *animal*, not
  Slack's *animal face*, for a handful of names: ``dog`` -> ``🐕`` (Slack means
  ``🐶``), and likewise ``cat``, ``mouse``, ``cow``, ``pig``, ``monkey``,
  ``horse``, ``rabbit``, ``tiger``. A real workspace serves those from
  ``emoji.list`` (step 2) and never reaches step 3; they are only wrong in the
  no-Slack-client case, and a literal character always wins.
"""

from __future__ import annotations

import io
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import httpx
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .types import HuddleError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SIZE",
    "EMOJI_FONT_CANDIDATES",
    "FONT_FIXED_SIZES",
    "MAX_DOWNLOAD_BYTES",
    "MAX_SIZE",
    "MAX_SOURCE_DIMENSION",
    "SHORTCODE_ALIASES",
    "EmojiRender",
    "EmojiSourceKind",
    "EmojiResolver",
    "emoji_image",
    "fetch_emoji_image",
    "find_emoji_font",
    "looks_like_unicode_emoji",
    "render_unicode_emoji",
    "resolve_emoji_image",
]

#: Default longest side of the returned image. 512 matches
#: :data:`huddle_drawings.image_trace.DEFAULT_MAX_DIMENSION`, so the tracer
#: neither upscales nor throws pixels away.
DEFAULT_SIZE = 512

#: Hard ceiling on the returned size. Upscaling a 64 px Slack PNG past this
#: invents nothing a 7 px pen could draw, and contour cost grows with the
#: perimeter in pixels.
MAX_SIZE = 1024

#: Candidate colour emoji fonts, **Noto first** (it is the one Pillow handles
#: reliably: Apple Color Emoji is ``sbix`` and Segoe UI Emoji needs COLRv0
#: support, both of which Pillow may refuse or render as flat outlines).
EMOJI_FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/usr/share/fonts/NotoColorEmoji.ttf",
    "/usr/local/share/fonts/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
    "/usr/share/fonts/opentype/openmoji/OpenMoji-Color.ttf",
    "/usr/share/fonts/truetype/openmoji/OpenMoji-color-colr0_svg.ttf",
    "/usr/share/fonts/truetype/twemoji/Twemoji.ttf",
    "/Library/Fonts/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "C:\\Windows\\Fonts\\seguiemj.ttf",
)

#: Pixel strikes to try when the font refuses the requested size. 109 is what
#: current ``NotoColorEmoji.ttf`` ships; the others are strikes seen in older
#: builds and in other CBDT fonts.
FONT_FIXED_SIZES: tuple[int, ...] = (109, 128, 136, 160, 96, 64, 32)

#: Refuse a download bigger than this. A Slack emoji is a few KB.
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024

#: Refuse a decoded image longer than this on either side, and more than
#: :data:`MAX_SOURCE_PIXELS` in total -- a decompression bomb dressed as an
#: emoji.
MAX_SOURCE_DIMENSION = 8192
MAX_SOURCE_PIXELS = 32_000_000

#: Codepoint ranges counted as "a pictograph the emoji font might draw".
#: Deliberately conservative: it must **reject** the near-misses
#: :func:`unicodedata.lookup` finds for Slack shortcodes, e.g. ``smile`` ->
#: U+2323 ``⌣`` SMILE (a maths symbol) and ``house`` -> U+2302 ``⌂`` HOUSE.
#: Those are not emoji, no emoji font draws them, and silently drawing a blank
#: is worse than falling through to :data:`SHORTCODE_ALIASES`.
_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x00A9, 0x00A9),  # (c)
    (0x00AE, 0x00AE),  # (r)
    (0x203C, 0x203C),  # !!
    (0x2049, 0x2049),  # !?
    (0x2122, 0x2122),  # TM
    (0x2139, 0x2139),  # information
    (0x2194, 0x2199),  # arrows (the emoji ones only)
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),  # watch, hourglass
    (0x2328, 0x2328),  # keyboard
    (0x23CF, 0x23CF),
    (0x23E9, 0x23FA),  # media controls, clocks
    (0x24C2, 0x24C2),  # circled M
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x27BF),  # misc symbols + dingbats
    (0x2934, 0x2935),
    (0x2B00, 0x2BFF),  # arrows, stars, big squares/circles
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1FAFF),  # every pictographic plane-1 block
)

#: Joiners, selectors and modifiers that are part of an emoji *sequence* but
#: are not pictographs on their own.
_SEQUENCE_CHARS = frozenset(
    {
        0x200D,  # ZWJ
        0x20E3,  # combining enclosing keycap
        0xFE0E,  # text presentation selector
        0xFE0F,  # emoji presentation selector
    }
)

#: Longest spec :func:`looks_like_unicode_emoji` will even consider. The
#: longest standard sequence (a subdivision flag) is 14 codepoints.
_MAX_EMOJI_CODEPOINTS = 32

#: Common Slack shortcodes whose Unicode name differs from Slack's, where the
#: mapping is unambiguous. **Deliberately not exhaustive** and not meant to
#: become so: Slack has ~1,800 standard shortcodes and mirroring that table
#: here would be a vendored dataset to keep in sync. These are the ones that
#: come up in a huddle, and that neither ``emoji.list`` (for workspaces that
#: render them from Slack's own sprite) nor :func:`unicodedata.lookup` finds.
#: Anything missing is one literal emoji character away from working.
SHORTCODE_ALIASES: dict[str, str] = {
    "tada": "\N{PARTY POPPER}",
    "+1": "\N{THUMBS UP SIGN}",
    "thumbsup": "\N{THUMBS UP SIGN}",
    "-1": "\N{THUMBS DOWN SIGN}",
    "thumbsdown": "\N{THUMBS DOWN SIGN}",
    "100": "\N{HUNDRED POINTS SYMBOL}",
    "joy": "\N{FACE WITH TEARS OF JOY}",
    "sob": "\N{LOUDLY CRYING FACE}",
    "smile": "\N{SMILING FACE WITH OPEN MOUTH AND SMILING EYES}",
    "smiley": "\N{SMILING FACE WITH OPEN MOUTH}",
    "grinning": "\N{GRINNING FACE}",
    "heart_eyes": "\N{SMILING FACE WITH HEART-SHAPED EYES}",
    "wink": "\N{WINKING FACE}",
    "scream": "\N{FACE SCREAMING IN FEAR}",
    "rage": "\N{POUTING FACE}",
    "heart": "\N{HEAVY BLACK HEART}",
    "pray": "\N{PERSON WITH FOLDED HANDS}",
    "clap": "\N{CLAPPING HANDS SIGN}",
    "wave": "\N{WAVING HAND SIGN}",
    "muscle": "\N{FLEXED BICEPS}",
    "ok_hand": "\N{OK HAND SIGN}",
    "raised_hands": "\N{PERSON RAISING BOTH HANDS IN CELEBRATION}",
    "eyes": "\N{EYES}",
    "white_check_mark": "\N{WHITE HEAVY CHECK MARK}",
    "x": "\N{CROSS MARK}",
    "warning": "\N{WARNING SIGN}",
    "zap": "\N{HIGH VOLTAGE SIGN}",
    "boom": "\N{COLLISION SYMBOL}",
    "star": "\N{WHITE MEDIUM STAR}",
    "poop": "\N{PILE OF POO}",
    "hankey": "\N{PILE OF POO}",
    "fire": "\N{FIRE}",
    "coffee": "\N{HOT BEVERAGE}",
    "pizza": "\N{SLICE OF PIZZA}",
    "beer": "\N{BEER MUG}",
    "cake": "\N{BIRTHDAY CAKE}",
    "gift": "\N{WRAPPED PRESENT}",
    "alien": "\N{EXTRATERRESTRIAL ALIEN}",
    "mag": "\N{LEFT-POINTING MAGNIFYING GLASS}",
    "see_no_evil": "\N{SEE-NO-EVIL MONKEY}",
}

#: Which branch produced the image.
#:
#: * ``"unicode"`` -- ``spec`` was a literal emoji character, rendered locally.
#: * ``"slack"`` -- resolved through ``emoji.list`` and downloaded.
#: * ``"unicodedata"`` -- the shortcode matched a Unicode character *name*.
#: * ``"alias"`` -- the shortcode came from :data:`SHORTCODE_ALIASES`.
EmojiSourceKind: TypeAlias = Literal["unicode", "slack", "unicodedata", "alias"]


class EmojiResolver(Protocol):
    """The one method :func:`emoji_image` needs from a Slack client.

    :class:`huddle_drawings.slack.SlackClient` satisfies this; so does a
    two-line fake in a test.
    """

    async def resolve_emoji(self, name: str, *, max_hops: int = 8) -> str:
        """Return an image URL for ``name``, following ``alias:`` chains."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class EmojiRender:
    """An emoji image plus where it came from.

    ``image`` is the opaque ``RGB`` image; the rest is provenance, which the
    CLI/TUI can show ("drew :tada: from the local font") and which the tests
    assert on. The same fields are also stashed in ``image.info`` under
    ``emoji_spec`` / ``emoji_source`` / ``emoji_detail``, so a caller that only
    holds the :class:`PIL.Image.Image` can still tell what happened.
    """

    #: Opaque ``RGB`` image, longest side ``min(size, MAX_SIZE)``.
    image: Image.Image
    #: Exactly what the caller asked for.
    spec: str
    #: Which branch won.
    kind: EmojiSourceKind
    #: One human-readable sentence about that branch.
    detail: str
    #: The literal character rendered, for the font-backed branches.
    character: str | None = None
    #: The URL downloaded, for ``"slack"``.
    url: str | None = None
    #: True when the download was an animated GIF and only frame 0 was used.
    animated: bool = False
    #: Every branch tried and rejected before the winner, newest last.
    attempts: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# classifying the spec
# --------------------------------------------------------------------------


def _in_emoji_range(codepoint: int) -> bool:
    return any(low <= codepoint <= high for low, high in _EMOJI_RANGES)


def looks_like_unicode_emoji(spec: str) -> bool:
    """True when ``spec`` is a literal emoji character or sequence.

    Accepts single pictographs (``🎉``), presentation-selector pairs
    (``❤️``), keycaps (``1️⃣``), regional-indicator flags (``🇯🇵``), ZWJ
    sequences (``👩‍👩‍👧``) and skin-tone modifiers.

    Rejects anything that is a plausible *name* instead -- ``"tada"``,
    ``"hello"``, ``"+1"``, ``"100"``, ``":squirrel:"`` -- and rejects ``""``.
    Note that a bare digit is only accepted as part of a keycap sequence, so
    ``"100"`` is a name and ``"1️⃣"`` is an emoji.
    """
    text = spec.strip()
    if not text or len(text) > _MAX_EMOJI_CODEPOINTS:
        return False
    keycap = "\u20e3" in text
    pictographic = False
    for char in text:
        code = ord(char)
        if _in_emoji_range(code):
            pictographic = True
            continue
        if code in _SEQUENCE_CHARS:
            continue
        if 0xE0020 <= code <= 0xE007F:  # tag characters: subdivision flags
            continue
        if keycap and (char.isdigit() or char in "#*"):
            continue
        return False
    return pictographic or keycap


def _emoji_name(spec: str) -> str:
    """Normalise a Slack-style spec into a bare lower-case name."""
    return spec.strip().strip(":").strip().lower()


def _codepoints(text: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in text)


def _unicodedata_character(name: str) -> str | None:
    """``party_popper`` -> ``🎉`` via :func:`unicodedata.lookup`, or ``None``.

    Slack names are lower-case with ``_`` (standard) or ``-`` (custom) where a
    Unicode character name has spaces, so both separators are expanded. The
    result is only returned if it is actually pictographic -- see
    :data:`_EMOJI_RANGES` for why that check is not optional.
    """
    candidate = name.replace("_", " ").replace("-", " ").strip().upper()
    if not candidate:
        return None
    try:
        char = unicodedata.lookup(candidate)
    except (KeyError, TypeError):
        return None
    if not looks_like_unicode_emoji(char):
        logger.debug(
            "unicodedata.lookup(%r) gave %s (%s), which is not a pictograph; "
            "ignoring it",
            candidate,
            char,
            _codepoints(char),
        )
        return None
    return char


# --------------------------------------------------------------------------
# fonts
# --------------------------------------------------------------------------


def find_emoji_font() -> Path | None:
    """First existing font in :data:`EMOJI_FONT_CANDIDATES`, or ``None``.

    Not cached, so a test can monkeypatch either this function or the
    candidate list.
    """
    for candidate in EMOJI_FONT_CANDIDATES:
        path = Path(candidate)
        try:
            if path.is_file():
                return path
        except OSError:  # unreadable mount, weird path
            continue
    return None


def _font_missing_error() -> HuddleError:
    return HuddleError(
        "no colour emoji font found, so a literal emoji character cannot be "
        "rendered. Install one (on Linux: the 'noto-fonts-emoji' / "
        "'fonts-noto-color-emoji' package, which puts NotoColorEmoji.ttf in "
        "/usr/share/fonts), pass font_path=..., or use a Slack emoji name "
        "(:tada:) so the image can be downloaded from the workspace instead. "
        "Looked in: " + ", ".join(EMOJI_FONT_CANDIDATES)
    )


def _open_font(path: Path, size: int) -> tuple[ImageFont.FreeTypeFont, int]:
    """Open ``path`` at ``size`` if it can, else at a fixed strike.

    Returns the font and the pixel size it was actually opened at. A CBDT
    bitmap font such as ``NotoColorEmoji.ttf`` raises ``OSError: invalid pixel
    size`` for every size but its one embedded strike, which is why this tries
    :data:`FONT_FIXED_SIZES` rather than giving up.
    """
    errors: list[str] = []
    for attempt in (size, *FONT_FIXED_SIZES):
        if attempt < 1:
            continue
        try:
            return ImageFont.truetype(str(path), attempt), attempt
        except OSError as exc:
            errors.append(f"{attempt}px: {exc}")
    raise HuddleError(
        f"{path} could not be opened as a font at any of the sizes tried "
        f"({'; '.join(errors)}). If it is a bitmap (CBDT/sbix) font, Pillow "
        "only renders it at an embedded strike size"
    )


def _normalise_size(size: int) -> int:
    if size < 1:
        raise ValueError("size must be >= 1")
    if size > MAX_SIZE:
        logger.debug(
            "clamping requested emoji size %d to %d: a 64px source has no "
            "more detail to give",
            size,
            MAX_SIZE,
        )
        return MAX_SIZE
    return size


def render_unicode_emoji(
    text: str,
    *,
    size: int = DEFAULT_SIZE,
    font_path: str | Path | None = None,
) -> Image.Image:
    """Render a literal emoji character from a local font.

    Returns an **RGBA** image (alpha preserved, cropped to the glyph) whose
    longest side is ``min(size, MAX_SIZE)``; :func:`emoji_image` is what
    flattens it onto a background.

    The font is opened at ``size`` if it will allow it and at a fixed strike
    otherwise, then the glyph is LANCZOS-upscaled -- see the module docstring.

    :raises HuddleError: if no emoji font can be found or opened, or if the
        font has no glyph for ``text`` (which is how the shortcode fallbacks
        tell a real hit from a near-miss: a missing glyph renders as nothing
        at all rather than as tofu, in ``NotoColorEmoji``).
    """
    if not text:
        raise HuddleError("nothing to render: empty emoji text")
    target = _normalise_size(size)
    chosen = Path(font_path) if font_path is not None else find_emoji_font()
    if chosen is None:
        raise _font_missing_error()
    if not chosen.is_file():
        raise HuddleError(f"emoji font {chosen} does not exist")

    font, native = _open_font(chosen, target)
    # Generous canvas: an unsupported ZWJ sequence degrades into one glyph per
    # codepoint laid out side by side, and we crop to the ink afterwards.
    margin = native
    width = margin * 2 + native * min(max(len(text), 1), 12)
    height = margin * 2 + native * 2
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text(
            (margin, margin),
            text,
            font=font,
            fill=(0, 0, 0, 255),
            embedded_color=True,
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - font-specific
        raise HuddleError(
            f"{chosen.name} failed to render {text!r} ({_codepoints(text)}): "
            f"{exc}"
        ) from exc

    box = canvas.getchannel("A").getbbox()
    if box is None:
        raise HuddleError(
            f"{chosen.name} has no glyph for {text!r} ({_codepoints(text)}): "
            "it rendered a blank image"
        )
    glyph = canvas.crop(box)
    logger.debug(
        "rendered %r (%s) from %s at %dpx -> %dx%d",
        text,
        _codepoints(text),
        chosen.name,
        native,
        *glyph.size,
    )
    return _scale_to(glyph, target)


# --------------------------------------------------------------------------
# downloading
# --------------------------------------------------------------------------


def _decode_image_bytes(data: bytes, url: str) -> tuple[Image.Image, bool]:
    """Decode ``data`` to RGBA, taking frame 0 of an animation.

    Returns ``(image, animated)``.
    """
    try:
        opened = Image.open(io.BytesIO(data))
        animated = bool(getattr(opened, "is_animated", False))
        if animated:
            logger.info(
                "%s is an animated %s with %d frames; drawing the first frame "
                "only",
                url,
                opened.format or "image",
                getattr(opened, "n_frames", 1),
            )
            opened.seek(0)
        width, height = opened.size
        if (
            width < 1
            or height < 1
            or max(width, height) > MAX_SOURCE_DIMENSION
            or width * height > MAX_SOURCE_PIXELS
        ):
            raise HuddleError(
                f"{url} is {width}x{height}, which is not a sane size for an "
                f"emoji (limit {MAX_SOURCE_DIMENSION}px a side)"
            )
        frame = opened.convert("RGBA")
    except HuddleError:
        raise
    except (OSError, ValueError, EOFError, Image.DecompressionBombError) as exc:
        raise HuddleError(
            f"{url} did not return a decodable image ({type(exc).__name__}: "
            f"{exc}). Slack emoji URLs serve PNG/GIF/JPEG; a login page or an "
            "error body will land here"
        ) from exc
    return frame, animated


async def fetch_emoji_image(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 15.0,
) -> Image.Image:
    """Download an emoji image and decode it to **RGBA**.

    An animated GIF is reduced to its first frame (logged at INFO). Alpha is
    kept; compositing is :func:`emoji_image`'s job.

    Args:
        url: an ``http(s)`` URL, normally ``https://emoji.slack-edge.com/...``.
        client: an :class:`httpx.AsyncClient` to borrow. Injectable so tests
            can hand in an ``httpx.MockTransport`` and never touch a network.
            When omitted a client is created and closed around the request.
        timeout: per-request timeout in seconds.

    :raises HuddleError: on a transport error, a non-2xx status, an empty or
        oversized body, or content that is not a decodable image. The URL is
        always in the message.
    """
    if not url or not url.lower().startswith(("http://", "https://")):
        raise HuddleError(f"not an http(s) emoji image URL: {url!r}")

    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        data = response.content
    except httpx.HTTPStatusError as exc:
        raise HuddleError(
            f"downloading {url} failed with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise HuddleError(f"downloading {url} failed: {exc}") from exc
    finally:
        if owned:
            await http.aclose()

    if not data:
        raise HuddleError(f"{url} returned an empty body")
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise HuddleError(
            f"{url} returned {len(data)} bytes, over the "
            f"{MAX_DOWNLOAD_BYTES}-byte limit for an emoji image"
        )
    frame, animated = _decode_image_bytes(data, url)
    # Provenance for callers that only get the image back.
    frame.info["emoji_animated"] = animated
    frame.info["emoji_url"] = url
    return frame


# --------------------------------------------------------------------------
# image plumbing
# --------------------------------------------------------------------------


def _scale_to(img: Image.Image, target: int) -> Image.Image:
    """Resize so the longest side is ``target``, preserving aspect (LANCZOS)."""
    longest = max(img.size)
    if longest == target or longest == 0:
        return img
    factor = target / float(longest)
    size = (max(1, round(img.width * factor)), max(1, round(img.height * factor)))
    if size == img.size:
        return img
    logger.debug(
        "%s %dx%d -> %dx%d (LANCZOS)",
        "upscaling" if factor > 1 else "downscaling",
        *img.size,
        *size,
    )
    return img.resize(size, Image.Resampling.LANCZOS)


def _trim_transparent(img: Image.Image, *, what: str) -> Image.Image:
    """Crop away fully transparent margins.

    Slack's PNGs are padded to a square, and a font glyph sits inside its em
    box; without this the tracer scales the padding and the emoji ends up small
    in the middle of the drawing area.
    """
    if img.mode != "RGBA":
        return img
    box = img.getchannel("A").getbbox()
    if box is None:
        raise HuddleError(
            f"{what} is fully transparent: there is nothing to draw"
        )
    if box == (0, 0, img.width, img.height):
        return img
    logger.debug("trimming transparent margin %s from %dx%d", box, *img.size)
    return img.crop(box)


def _normalise_background(background: tuple[int, int, int] | str) -> tuple[int, int, int]:
    if isinstance(background, str):
        rgb = ImageColor.getrgb(background)
        return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    values = tuple(int(v) for v in background)
    if len(values) != 3 or not all(0 <= v <= 255 for v in values):
        raise ValueError(
            f"background must be three 0..255 channels or a colour name, got "
            f"{background!r}"
        )
    return (values[0], values[1], values[2])


def _composite(img: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    """Flatten onto ``background`` and return ``RGB``."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    plate = Image.new("RGBA", img.size, (*background, 255))
    return Image.alpha_composite(plate, img).convert("RGB")


def _finalise(
    img: Image.Image,
    *,
    size: int,
    background: tuple[int, int, int],
    what: str,
) -> Image.Image:
    """Trim, scale, flatten: the order matters (see the module docstring)."""
    trimmed = _trim_transparent(img, what=what)
    return _composite(_scale_to(trimmed, size), background)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


async def resolve_emoji_image(
    spec: str,
    *,
    slack: EmojiResolver | None = None,
    size: int = DEFAULT_SIZE,
    background: tuple[int, int, int] | str = (255, 255, 255),
    client: httpx.AsyncClient | None = None,
    font_path: str | Path | None = None,
    timeout: float = 15.0,
) -> EmojiRender:
    """Resolve ``spec`` to an image **and say where it came from**.

    :func:`emoji_image` is the same thing without the provenance; see it for
    the resolution order and the arguments.
    """
    target = _normalise_size(size)
    plate = _normalise_background(background)
    text = spec.strip()
    if not text:
        raise HuddleError("empty emoji spec: pass an emoji character or a name")

    # 1. a literal character (or sequence) -- covers all of Unicode with no
    #    name table and no network.
    if looks_like_unicode_emoji(text):
        glyph = render_unicode_emoji(text, size=target, font_path=font_path)
        return _render(
            _finalise(
                glyph, size=target, background=plate, what=f"the glyph for {text!r}"
            ),
            spec=spec,
            kind="unicode",
            detail=f"rendered the literal character {text} ({_codepoints(text)})",
            character=text,
        )

    name = _emoji_name(text)
    if not name:
        raise HuddleError(f"{spec!r} is not an emoji character or a name")
    attempts: list[str] = []

    # 2. a Slack emoji name: custom uploads *and* most of the standard set.
    if slack is not None:
        url: str | None = None
        try:
            url = await slack.resolve_emoji(name)
        except HuddleError as exc:
            # Normal, not exceptional: emoji.list does not cover the emoji
            # Slack renders from its own sprite (:tada: is absent from a real
            # workspace's 65k-entry map).
            attempts.append(f"this workspace's emoji.list ({exc})")
            logger.debug("%s not in emoji.list: %s", name, exc)
        if url is not None:
            downloaded = await fetch_emoji_image(
                url, client=client, timeout=timeout
            )
            animated = bool(downloaded.info.get("emoji_animated", False))
            return _render(
                _finalise(
                    downloaded,
                    size=target,
                    background=plate,
                    what=f":{name}: from {url}",
                ),
                spec=spec,
                kind="slack",
                detail=f"downloaded :{name}: from {url}",
                url=url,
                animated=animated,
                attempts=tuple(attempts),
            )
    else:
        attempts.append(
            "this workspace's emoji.list (skipped: no Slack client was given)"
        )

    # 3. the shortcode *is* the Unicode character name, often enough.
    character = _unicodedata_character(name)
    if character is not None:
        try:
            glyph = render_unicode_emoji(character, size=target, font_path=font_path)
        except HuddleError as exc:
            attempts.append(
                f"unicodedata.lookup({name!r}) -> {character} (but it did not "
                f"render: {exc})"
            )
        else:
            return _render(
                _finalise(
                    glyph,
                    size=target,
                    background=plate,
                    what=f"the glyph for :{name}:",
                ),
                spec=spec,
                kind="unicodedata",
                detail=(
                    f"matched the Unicode character name "
                    f"{name.replace('_', ' ').replace('-', ' ').upper()!r} -> "
                    f"{character} and rendered it from the local font"
                ),
                character=character,
                attempts=tuple(attempts),
            )
    else:
        attempts.append(f"unicodedata.lookup for {name!r} (no pictographic match)")

    # 4. the small curated table, for the handful where Slack's name and the
    #    Unicode name disagree.
    curated = SHORTCODE_ALIASES.get(name)
    if curated is not None:
        try:
            glyph = render_unicode_emoji(curated, size=target, font_path=font_path)
        except HuddleError as exc:
            attempts.append(
                f"the curated alias :{name}: -> {curated} (but it did not "
                f"render: {exc})"
            )
        else:
            return _render(
                _finalise(
                    glyph,
                    size=target,
                    background=plate,
                    what=f"the glyph for :{name}:",
                ),
                spec=spec,
                kind="alias",
                detail=(
                    f"used the curated alias :{name}: -> {curated} and "
                    "rendered it from the local font"
                ),
                character=curated,
                attempts=tuple(attempts),
            )
    else:
        attempts.append(
            f"this module's curated shortcode table ({len(SHORTCODE_ALIASES)} "
            f"entries, no :{name}:)"
        )

    # 5. nothing worked. Say what was tried; the literal character always works
    #    if a font is installed.
    raise HuddleError(
        f"could not turn :{name}: into an image. Tried: "
        + "; ".join(attempts)
        + ". Pass the literal emoji character instead of the name (e.g. "
        "'\N{PARTY POPPER}' rather than ':tada:') -- that path needs no name "
        "table at all"
    )


def _render(
    image: Image.Image,
    *,
    spec: str,
    kind: EmojiSourceKind,
    detail: str,
    character: str | None = None,
    url: str | None = None,
    animated: bool = False,
    attempts: tuple[str, ...] = (),
) -> EmojiRender:
    image.info["emoji_spec"] = spec
    image.info["emoji_source"] = kind
    image.info["emoji_detail"] = detail
    logger.info("emoji %r via %s: %s", spec, kind, detail)
    return EmojiRender(
        image=image,
        spec=spec,
        kind=kind,
        detail=detail,
        character=character,
        url=url,
        animated=animated,
        attempts=attempts,
    )


async def emoji_image(
    spec: str,
    *,
    slack: EmojiResolver | None = None,
    size: int = DEFAULT_SIZE,
    background: tuple[int, int, int] | str = (255, 255, 255),
    client: httpx.AsyncClient | None = None,
    font_path: str | Path | None = None,
    timeout: float = 15.0,
) -> Image.Image:
    """Turn an emoji character or a Slack emoji name into a traceable image.

    The returned image is opaque ``RGB`` with its longest side
    ``min(size, MAX_SIZE)``, ready to hand straight to
    :func:`huddle_drawings.image_trace.trace_image_colors` (or ``trace_image``
    for a one-colour outline). ``image.info`` carries ``emoji_spec``,
    ``emoji_source`` and ``emoji_detail``; call :func:`resolve_emoji_image` for
    the same thing as a structured :class:`EmojiRender`.

    Resolution order, first hit wins:

    1. ``spec`` is a **literal emoji character or sequence** (see
       :func:`looks_like_unicode_emoji`): rendered from the local colour emoji
       font. No network, no name table, works for every ZWJ family and flag.
    2. otherwise it is a **Slack emoji name** (surrounding colons optional):
       ``slack.resolve_emoji(name)`` then download. This covers custom uploads
       *and* most of the standard set, since Slack serves the standard emoji as
       PNGs too.
    3. the name is **not in this workspace's** ``emoji.list`` -- which is
       normal, Slack renders part of the standard set from its own sprite --
       so :func:`unicodedata.lookup` is tried on the name with ``_``/``-`` as
       spaces, upper-cased (``party_popper`` -> ``PARTY POPPER`` -> 🎉), and
       the result rendered from the font. A match that is not pictographic is
       rejected (``smile`` would otherwise find U+2323 ``⌣``).
    4. then :data:`SHORTCODE_ALIASES`, a short curated table for the common
       shortcodes whose Slack name differs from the Unicode name (``tada``,
       ``+1``, ``100``, ``joy``...). Deliberately not exhaustive.
    5. otherwise :class:`~huddle_drawings.types.HuddleError`, listing what was
       tried and suggesting the literal character.

    Args:
        spec: an emoji character/sequence, or a Slack name with or without
            colons (``"tada"``, ``":tada:"``, ``"my-custom-thing"``).
        slack: anything with ``async resolve_emoji(name) -> url`` -- normally a
            :class:`huddle_drawings.slack.SlackClient`. ``None`` skips step 2,
            so no-Slack use (``emoji_image("🎉")``) needs no credentials.
        size: longest side of the result, clamped to :data:`MAX_SIZE`. Small
            sources are LANCZOS-upscaled so contour tracing has pixels to work
            with.
        background: what alpha is composited onto; ``(255, 255, 255)`` or a
            Pillow colour name. **Not** optional in effect -- the tracer needs
            opaque pixels or a transparent background traces as a solid block.
        client: :class:`httpx.AsyncClient` to borrow for the download.
        font_path: override the font instead of searching
            :data:`EMOJI_FONT_CANDIDATES`.
        timeout: per-request download timeout in seconds.

    :raises HuddleError: if the emoji cannot be resolved, the download fails or
        is not an image, no emoji font is installed for a font-backed path, or
        the resulting image is fully transparent.
    """
    return (
        await resolve_emoji_image(
            spec,
            slack=slack,
            size=size,
            background=background,
            client=client,
            font_path=font_path,
            timeout=timeout,
        )
    ).image
