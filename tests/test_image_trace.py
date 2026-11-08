"""Tests for the image tracer: contours, budgets, placement, previews.

Every fixture image is generated in code with Pillow into ``tmp_path`` -- there
is nothing to fetch, and nothing outside the test run to break.

The invariant that matters most here is the **segment budget**: one segment is
one ``draw`` message against Chime's ~150/s per-meeting bucket, so
``max_segments`` is tested as a hard ceiling across every mode and a spread of
budgets, including absurd ones.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from huddle_drawings import geometry as g
from huddle_drawings import image_trace as it

MODES = ("outline", "threshold", "fill", "auto")
BUDGETS = (4, 7, 13, 40, 120, 600)


# --------------------------------------------------------------------------
# fixtures: every test image is drawn here, in code
# --------------------------------------------------------------------------


def _save(img: Image.Image, path: Path) -> Path:
    img.save(path)
    return path


@pytest.fixture
def square(tmp_path: Path) -> Path:
    """A black square on white: the simplest thing with four corners."""
    img = Image.new("L", (128, 128), 255)
    ImageDraw.Draw(img).rectangle([32, 32, 95, 95], fill=0)
    return _save(img, tmp_path / "square.png")


@pytest.fixture
def circle(tmp_path: Path) -> Path:
    img = Image.new("L", (200, 200), 255)
    ImageDraw.Draw(img).ellipse([20, 20, 179, 179], fill=0)
    return _save(img, tmp_path / "circle.png")


@pytest.fixture
def diagonal(tmp_path: Path) -> Path:
    """A thin diagonal line: nothing but staircase to simplify away."""
    img = Image.new("L", (160, 160), 255)
    ImageDraw.Draw(img).line([10, 10, 149, 149], fill=0, width=2)
    return _save(img, tmp_path / "diagonal.png")


@pytest.fixture
def gradient(tmp_path: Path) -> Path:
    """A smooth left-to-right ramp: no edges anywhere."""
    img = Image.linear_gradient("L").resize((128, 128)).rotate(90)
    return _save(img, tmp_path / "gradient.png")


@pytest.fixture
def blank(tmp_path: Path) -> Path:
    return _save(Image.new("L", (64, 64), 255), tmp_path / "blank.png")


@pytest.fixture
def one_pixel(tmp_path: Path) -> Path:
    return _save(Image.new("L", (1, 1), 0), tmp_path / "pixel.png")


@pytest.fixture
def transparent(tmp_path: Path) -> Path:
    """RGBA: an opaque black square on a fully transparent background."""
    img = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([20, 20, 99, 99], fill=(0, 0, 0, 255))
    return _save(img, tmp_path / "rgba.png")


@pytest.fixture
def wide(tmp_path: Path) -> Path:
    """A 2:1 image holding a 160x60 black rectangle (also 8:3)."""
    img = Image.new("L", (200, 100), 255)
    ImageDraw.Draw(img).rectangle([20, 20, 179, 79], fill=0)
    return _save(img, tmp_path / "wide.png")


@pytest.fixture
def ring_and_specks(tmp_path: Path) -> Path:
    """One big ring plus 40 three-pixel specks of noise."""
    img = Image.new("L", (300, 300), 255)
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 269, 269], outline=0, width=4)
    for i in range(40):
        x, y = 10 + (i * 7) % 280, 5 + (i * 13) % 20
        draw.rectangle([x, y, x + 2, y + 2], fill=0)
    return _save(img, tmp_path / "noisy.png")


@pytest.fixture
def all_images(
    square: Path,
    circle: Path,
    diagonal: Path,
    gradient: Path,
    blank: Path,
    one_pixel: Path,
    transparent: Path,
    wide: Path,
    ring_and_specks: Path,
) -> list[Path]:
    return [
        square,
        circle,
        diagonal,
        gradient,
        blank,
        one_pixel,
        transparent,
        wide,
        ring_and_specks,
    ]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def points(result: it.TraceResult) -> list[g.Point]:
    """Every coordinate that goes on the wire, control points included."""
    return [p for s in result.segments for p in s.points]


def bbox(result: it.TraceResult) -> g.Rect:
    """Bounding box of the *drawn curve*, by sampling the Béziers.

    Not the control points' box: a Catmull-Rom control point can sit well
    outside the curve it shapes (and a hairpin's deliberately sits 4/3 of the
    way past its tip), so control points say nothing about where the ink is.
    """
    sampled = [
        g.bezier_point(s, i / 16) for s in result.segments for i in range(17)
    ]
    return g.Rect.bounding(sampled)


def inside(rect: g.Rect, result: it.TraceResult, tol: float = 1e-9) -> bool:
    r = rect.normalized()
    return all(
        r.x0 - tol <= p.x <= r.x1 + tol and r.y0 - tol <= p.y <= r.y1 + tol
        for p in points(result)
    )


# --------------------------------------------------------------------------
# contours_from_mask
# --------------------------------------------------------------------------


def test_contours_from_mask_traces_a_square_block() -> None:
    import numpy as np

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    contours = it.contours_from_mask(mask)
    assert len(contours) == 1
    loop = contours[0]
    # 6x6 block of pixels -> a 24-corner rectilinear loop, in corner
    # coordinates normalized by the longer side (10).
    assert len(loop) == 24
    box = g.Rect.bounding(loop)
    assert (box.x0, box.y0, box.x1, box.y1) == (0.2, 0.2, 0.8, 0.8)


def test_contours_from_mask_finds_holes_as_their_own_loops() -> None:
    import numpy as np

    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 3:17] = True
    mask[8:12, 8:12] = False
    contours = it.contours_from_mask(mask)
    assert len(contours) == 2, "the outside and the hole are both boundaries"
    outer, inner = sorted(contours, key=it._contour_length, reverse=True)
    assert it._contour_length(outer) > it._contour_length(inner)


def test_contours_from_mask_empty_and_shape_errors() -> None:
    import numpy as np

    assert it.contours_from_mask(np.zeros((5, 5), dtype=bool)) == []
    with pytest.raises(ValueError):
        it.contours_from_mask(np.zeros((2, 2, 2), dtype=bool))


# --------------------------------------------------------------------------
# the square: shape, cost and placement
# --------------------------------------------------------------------------


def test_square_traces_to_roughly_four_corners(square: Path) -> None:
    result = it.trace_image(square, mode="threshold")
    assert result.contours == 1
    # One closed curve through the four corners, plus the loop's start point
    # and whatever rounding the blur leaves: a handful of messages, not dozens.
    assert 4 <= len(result) <= 8, result.describe()
    assert result.source_size == (128, 128)
    assert result.mode == "threshold"


def test_square_segments_stay_inside_the_requested_rect(square: Path) -> None:
    rect = g.Rect(0.6, 0.1, 0.95, 0.45)
    for mode in MODES:
        result = it.trace_image(square, rect, mode=mode)
        assert result.segments, mode
        assert inside(rect, result), f"{mode} escaped the rect"
        assert result.rect == rect


def test_square_fills_the_rect_it_is_given(square: Path) -> None:
    rect = g.Rect(0.25, 0.25, 0.75, 0.75)
    box = bbox(it.trace_image(square, rect, mode="threshold"))
    # The square occupies half the source image, so it should land well inside
    # the rect but still be scaled up to use most of it.
    assert box.width > 0.3 * rect.width
    assert box.height > 0.3 * rect.height


# --------------------------------------------------------------------------
# normalization: everything comes back in 0..1
# --------------------------------------------------------------------------


def test_all_coordinates_are_normalized(all_images: list[Path]) -> None:
    unit = g.unit_square()
    for path in all_images:
        for mode in MODES:
            result = it.trace_image(path, mode=mode)
            assert inside(unit, result), f"{path.name}/{mode} left 0..1"


def test_coordinates_stay_in_0_1_for_odd_rects(circle: Path) -> None:
    for rect in (
        g.Rect(0.0, 0.0, 1.0, 1.0),
        g.Rect(0.0, 0.0, 0.1, 1.0),
        g.Rect(0.9, 0.9, 1.0, 1.0),
        g.Rect(0.8, 0.2, 0.2, 0.8),  # deliberately reversed
    ):
        for mode in MODES:
            result = it.trace_image(circle, rect, mode=mode)
            assert inside(g.unit_square(), result)
            assert inside(rect, result), (rect, mode)


# --------------------------------------------------------------------------
# the budget is hard
# --------------------------------------------------------------------------


def test_max_segments_is_never_exceeded(all_images: list[Path]) -> None:
    for path in all_images:
        for mode in MODES:
            for budget in BUDGETS:
                result = it.trace_image(path, mode=mode, max_segments=budget)
                assert len(result) <= budget, (
                    f"{path.name}/{mode}/{budget}: {result.describe()}"
                )
                assert result.max_segments == budget


def test_absurdly_small_budgets_still_draw_something(square: Path) -> None:
    for budget in (1, 2, 3, 4, 5):
        for mode in MODES:
            result = it.trace_image(square, mode=mode, max_segments=budget)
            assert len(result) <= budget
            assert result.segments, f"{mode} at budget {budget} drew nothing"


def test_zero_budget_draws_nothing_and_says_so(circle: Path) -> None:
    result = it.trace_image(circle, mode="threshold", max_segments=0)
    assert result.segments == []
    assert result.contours_dropped >= 1


def test_budget_degrades_by_simplifying_then_dropping(
    ring_and_specks: Path,
) -> None:
    generous = it.trace_image(
        ring_and_specks, mode="threshold", max_segments=2000,
        min_contour_length=0.0,
    )
    tight = it.trace_image(
        ring_and_specks, mode="threshold", max_segments=40,
        min_contour_length=0.0,
    )
    assert len(generous) > 40 >= len(tight)
    # Degradation is reported, not silent.
    assert tight.simplify_epsilon > generous.simplify_epsilon
    assert tight.contours_dropped > 0
    assert tight.contours < generous.contours
    assert tight.contours_found == generous.contours_found


def test_budget_keeps_the_longest_contours_not_the_first_ones(
    ring_and_specks: Path,
) -> None:
    """A tight budget must keep the ring, not 40 specks that happen to be first.

    The specks sit along the top of the image, so a tracer that truncated in
    scan order would return only them.
    """
    tight = it.trace_image(
        ring_and_specks, mode="threshold", max_segments=12,
        min_contour_length=0.0,
    )
    box = bbox(tight)
    assert box.height > 0.5, f"only the top of the image was drawn: {box}"


def test_monotonic_in_the_budget(circle: Path) -> None:
    counts = [
        len(it.trace_image(circle, mode="threshold", max_segments=b))
        for b in BUDGETS
    ]
    assert counts == sorted(counts), counts


def test_over_budget_is_logged(
    ring_and_specks: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="huddle_drawings.image_trace"):
        it.trace_image(
            ring_and_specks, mode="threshold", max_segments=20,
            min_contour_length=0.0,
        )
    assert any("budget" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# blank / degenerate input
# --------------------------------------------------------------------------


def test_blank_image_yields_no_segments(blank: Path) -> None:
    for mode in MODES:
        result = it.trace_image(blank, mode=mode)
        assert result.segments == []
        assert result.contours == 0
        assert result.estimated_seconds() == 0.0
        assert not result


def test_blank_image_is_blank_inverted_too(blank: Path) -> None:
    for mode in ("threshold", "fill"):
        assert it.trace_image(blank, mode=mode, invert=True).segments == []


def test_one_pixel_image_does_not_crash(one_pixel: Path) -> None:
    for mode in MODES:
        result = it.trace_image(one_pixel, mode=mode)
        assert result.source_size == (1, 1)
        assert result.traced_size == (1, 1)
        assert result.segments == []  # a uniform image has no regions


def test_smooth_gradient_has_no_edges(gradient: Path) -> None:
    assert it.trace_image(gradient, mode="outline").segments == []
    # ... but it does have a threshold, so threshold mode draws the split.
    assert it.trace_image(gradient, mode="threshold").segments


# --------------------------------------------------------------------------
# knobs
# --------------------------------------------------------------------------


def test_min_contour_length_filters_noise(ring_and_specks: Path) -> None:
    keep_all = it.trace_image(
        ring_and_specks, mode="threshold", min_contour_length=0.0,
        max_segments=2000,
    )
    filtered = it.trace_image(
        ring_and_specks, mode="threshold", min_contour_length=0.2,
        max_segments=2000,
    )
    assert keep_all.contours_found == filtered.contours_found
    assert keep_all.contours_filtered == 0
    assert filtered.contours_filtered >= 40
    assert filtered.contours == 2, "only the ring's two boundaries survive"
    assert len(filtered) < len(keep_all)


def test_invert_flips_which_side_is_traced(square: Path) -> None:
    """Inverting traces the light region -- minus the image frame.

    The light background's boundary is the square's edge *plus* the border of
    the image itself. That border is not in the picture and drawing it puts a
    box around the whole share, so it is rejected (``frame_coverage``) and
    reported in ``frames_dropped``. What is left is the square's boundary,
    which is what the uninverted trace draws too.
    """
    normal = it.trace_image(square, mode="threshold")
    flipped = it.trace_image(square, mode="threshold", invert=True)
    assert normal.contours == 1
    assert normal.frames_dropped == 0
    assert flipped.contours == 1
    assert flipped.frames_dropped == 1
    assert flipped.contours_found == 2, "the frame was found, then rejected"
    assert bbox(normal).width < 0.7, "just the square"
    assert bbox(flipped).width < 0.7, "just the square, the frame is gone"


def test_the_image_frame_can_be_kept_on_request(square: Path) -> None:
    kept = it.trace_image(
        square, mode="threshold", invert=True, frame_coverage=1.0
    )
    assert kept.contours == 2
    assert kept.frames_dropped == 0
    assert bbox(kept).width > 0.85, "the square plus the image frame"
    assert "image frame" in it.trace_image(
        square, mode="threshold", invert=True
    ).describe()


def test_a_big_round_shape_is_not_mistaken_for_the_frame(tmp_path: Path) -> None:
    """A circle that fills the image is a feature; only near-rectangles are frames."""
    img = Image.new("L", (200, 200), 255)
    ImageDraw.Draw(img).ellipse([0, 0, 199, 199], fill=0)
    path = _save(img, tmp_path / "bigcircle.png")
    result = it.trace_image(path, mode="threshold")
    assert result.frames_dropped == 0
    assert result.contours == 1
    assert bbox(result).width > 0.9


def test_invert_flips_which_side_is_filled(square: Path) -> None:
    normal = it.trace_image(square, mode="fill")
    flipped = it.trace_image(square, mode="fill", invert=True)
    assert bbox(flipped).width > bbox(normal).width * 1.2
    assert len(flipped) > len(normal), "the background is the bigger area"


def test_invert_is_a_documented_no_op_for_edges(square: Path) -> None:
    normal = it.trace_image(square, mode="outline")
    flipped = it.trace_image(square, mode="outline", invert=True)
    assert normal.segments == flipped.segments


def test_threshold_controls_the_split(tmp_path: Path) -> None:
    img = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(img).rectangle([10, 10, 49, 89], fill=0)  # black
    ImageDraw.Draw(img).rectangle([55, 10, 89, 89], fill=160)  # mid grey
    path = _save(img, tmp_path / "two_tones.png")
    dark_only = it.trace_image(path, mode="threshold", threshold=0.25)
    both = it.trace_image(path, mode="threshold", threshold=0.75)
    assert dark_only.contours == 1
    assert both.contours == 2
    assert dark_only.threshold == pytest.approx(0.25 * 255)


def test_threshold_must_be_a_fraction(square: Path) -> None:
    with pytest.raises(ValueError):
        it.trace_image(square, mode="threshold", threshold=200.0)


def test_simplify_trades_fidelity_for_messages(circle: Path) -> None:
    fine = it.trace_image(circle, mode="threshold", simplify=0.001,
                          max_segments=2000)
    coarse = it.trace_image(circle, mode="threshold", simplify=0.04,
                            max_segments=2000)
    assert len(fine) > len(coarse) >= 3


def test_fill_row_height_trades_density_for_messages(square: Path) -> None:
    dense = it.trace_image(square, mode="fill", fill_row_height=0.005,
                           max_segments=2000)
    sparse = it.trace_image(square, mode="fill", fill_row_height=0.05,
                            max_segments=2000)
    assert len(dense) > len(sparse) >= 1
    assert dense.fill_row_height == pytest.approx(0.005)


def test_fill_uses_hairpins_that_run_out_and_back(square: Path) -> None:
    """One message per band two pens tall: out along one row, back the next."""
    result = it.trace_image(square, mode="fill")
    assert result.segments
    for segment in result.segments:
        assert segment.a.y == pytest.approx(segment.b.y)
        assert segment.c.y == pytest.approx(segment.d.y)
        assert segment.d.y > segment.a.y  # the return row is below
        assert segment.a.x == pytest.approx(segment.d.x)  # ends above its start


def test_fill_loses_features_thinner_than_a_band(
    diagonal: Path, ring_and_specks: Path
) -> None:
    """A documented limitation, pinned so it cannot change silently.

    Fill mode asks whether half of a band's pixel rows are ink. A 2 px
    diagonal never is, so it fills to nothing; a 4 px ring only is where it
    runs horizontally, so a fill keeps its top and bottom arcs and loses the
    sides. Line art wants ``"outline"`` or ``"threshold"``.
    """
    assert it.trace_image(diagonal, mode="fill").segments == []
    assert it.trace_image(diagonal, mode="threshold").segments

    ring_fill = it.trace_image(ring_and_specks, mode="fill")
    ring_outline = it.trace_image(ring_and_specks, mode="threshold")
    assert 0 < len(ring_fill) < len(ring_outline) / 3
    # What survives is the top and bottom arcs, where the ring runs
    # horizontally and a band really is half ink; the left and right sides are
    # gone, so the fill is much narrower than it is tall.
    box = bbox(ring_fill)
    assert box.width < 0.5 * box.height


def test_unknown_mode_and_bad_arguments_raise(square: Path) -> None:
    with pytest.raises(ValueError):
        it.trace_image(square, mode="rotoscope")
    with pytest.raises(ValueError):
        it.trace_image(square, max_segments=-1)
    with pytest.raises(ValueError):
        it.trace_image(square, simplify=-0.1)
    with pytest.raises(ValueError):
        it.trace_image(square, max_dimension=0)


def test_edges_alias_matches_outline(square: Path) -> None:
    assert it.trace_image(square, mode="edges").segments == it.trace_image(
        square, mode="outline"
    ).segments


# --------------------------------------------------------------------------
# decoding: EXIF, alpha, downscaling
# --------------------------------------------------------------------------


def test_rgba_transparency_is_composited_onto_white(transparent: Path) -> None:
    """Transparent must read as background, not as ink."""
    result = it.trace_image(transparent, mode="threshold")
    assert result.contours == 1
    box = bbox(result)
    # The opaque square covers 80 of 120 px; the traced contour is that square,
    # not the whole frame.
    assert 0.5 < box.width < 0.95


def test_large_images_are_downscaled_before_tracing(tmp_path: Path) -> None:
    img = Image.new("L", (1800, 1200), 255)
    ImageDraw.Draw(img).ellipse([200, 200, 1599, 999], fill=0)
    path = _save(img, tmp_path / "big.png")
    result = it.trace_image(path, mode="threshold")
    assert result.source_size == (1800, 1200)
    assert max(result.traced_size) == it.DEFAULT_MAX_DIMENSION
    assert len(result) < 200, "a big ellipse must not cost hundreds of messages"


def test_max_dimension_bounds_the_work(circle: Path) -> None:
    small = it.trace_image(circle, mode="threshold", max_dimension=64)
    assert max(small.traced_size) == 64
    assert small.source_size == (200, 200)


def test_exif_orientation_is_honoured(tmp_path: Path) -> None:
    """Orientation 6 means "rotate 90 CW on display"; the trace must agree."""
    img = Image.new("L", (120, 60), 255)
    ImageDraw.Draw(img).rectangle([0, 0, 29, 59], fill=0)  # black left third
    exif = img.getexif()
    exif[0x0112] = 6
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif)

    result = it.trace_image(path, mode="threshold")
    assert result.source_size == (60, 120), "width and height swapped"
    box = bbox(result)
    assert box.height < 0.5, "the black band should end up along the top"


def test_accepts_an_open_pil_image(square: Path) -> None:
    with Image.open(square) as img:
        result = it.trace_image(img, mode="threshold")
    assert result.segments


# --------------------------------------------------------------------------
# placement and aspect ratio
# --------------------------------------------------------------------------


def test_aspect_ratio_is_preserved_for_a_non_square_image(wide: Path) -> None:
    result = it.trace_image(wide, mode="threshold")
    box = bbox(result)
    # The traced rectangle is 160x60 source pixels.
    assert box.width / box.height == pytest.approx(160 / 60, rel=0.05)


def test_aspect_can_be_stretched_on_request(wide: Path) -> None:
    preserved = bbox(it.trace_image(wide, mode="threshold"))
    stretched = bbox(
        it.trace_image(wide, mode="threshold", preserve_aspect=False)
    )
    assert stretched.height > preserved.height * 1.5, "fills the rect's height"
    assert stretched.width == pytest.approx(preserved.width, rel=0.01)
    assert stretched.width / stretched.height < 0.7 * (160 / 60)


def test_aspect_ratio_is_preserved_inside_a_rect(wide: Path) -> None:
    rect = g.Rect(0.1, 0.1, 0.9, 0.9)
    box = bbox(it.trace_image(wide, rect, mode="threshold"))
    assert box.width / box.height == pytest.approx(160 / 60, rel=0.05)
    assert box.height < rect.height, "centred on the slack axis"


# --------------------------------------------------------------------------
# TraceResult
# --------------------------------------------------------------------------


def test_estimated_seconds_scales_with_segment_count() -> None:
    def result(n: int) -> it.TraceResult:
        segment = g.Segment.line((0.0, 0.0), (0.1, 0.1))
        return it.TraceResult(
            segments=[segment] * n, contours=1, source_size=(10, 10)
        )

    assert result(0).estimated_seconds() == 0.0
    assert result(150).estimated_seconds() == pytest.approx(1.0)
    assert result(300).estimated_seconds() == pytest.approx(2.0)
    assert result(2000).estimated_seconds() == pytest.approx(13.33, rel=1e-3)
    # The default rate is the measured per-meeting budget, shared by everyone.
    assert result(75).estimated_seconds(rate=75.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        result(10).estimated_seconds(rate=0.0)


def test_estimated_seconds_tracks_the_real_trace(circle: Path) -> None:
    result = it.trace_image(circle, mode="threshold")
    assert result.estimated_seconds() == pytest.approx(len(result) / 150.0)


def test_result_describe_mentions_cost_and_losses(ring_and_specks: Path) -> None:
    text = it.trace_image(
        ring_and_specks, mode="threshold", max_segments=20,
        min_contour_length=0.0,
    ).describe()
    assert "segments" in text
    assert "dropped for budget" in text
    assert "s at 150/s" in text


def test_result_len_and_bool(square: Path) -> None:
    result = it.trace_image(square, mode="threshold")
    assert len(result) == len(result.segments)
    assert result


# --------------------------------------------------------------------------
# auto
# --------------------------------------------------------------------------


def test_auto_uses_threshold_for_a_high_contrast_image(square: Path) -> None:
    result = it.trace_image(square, mode="auto")
    assert result.mode == "threshold"
    assert result.segments == it.trace_image(square, mode="threshold").segments


def test_auto_falls_back_to_edges_when_thresholding_is_degenerate(
    tmp_path: Path,
) -> None:
    """A near-solid image: the threshold mask is almost everything, so the only
    'region boundary' would be the frame. Auto should try edges instead."""
    img = Image.new("L", (160, 160), 20)
    ImageDraw.Draw(img).rectangle([60, 60, 99, 99], fill=70)
    path = _save(img, tmp_path / "dark.png")
    result = it.trace_image(path, mode="auto")
    assert result.mode == "outline"


def test_auto_on_a_blank_image_returns_empty(blank: Path) -> None:
    result = it.trace_image(blank, mode="auto")
    assert result.segments == []


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------


def test_preview_ascii_is_stable_and_rectangular(square: Path) -> None:
    result = it.trace_image(square, mode="threshold")
    art = it.preview_ascii(result, 64)
    assert art == it.preview_ascii(result, 64), "same input, same art"
    lines = art.split("\n")
    assert len(lines) == 32
    assert {len(line) for line in lines} == {64}
    assert set("".join(lines)) <= {"#", "."}
    assert "#" in art


def test_preview_ascii_shows_the_shape_and_its_placement(square: Path) -> None:
    rect = g.Rect(0.0, 0.0, 0.5, 0.5)
    art = it.preview_ascii(it.trace_image(square, rect, mode="threshold"), 40)
    lines = art.split("\n")
    assert "#" in "".join(lines[:10]), "top-left quadrant should be drawn"
    assert "#" not in "".join(lines[-5:]), "bottom should be empty"
    assert all("#" not in line[-5:] for line in lines)


def test_preview_ascii_handles_an_empty_trace(blank: Path) -> None:
    art = it.preview_ascii(it.trace_image(blank, mode="threshold"), 20)
    assert set(art) == {".", "\n"}


def test_preview_ascii_accepts_bare_segments() -> None:
    art = it.preview_ascii([g.Segment.degenerate((0.5, 0.5))], 10)
    assert art.count("#") == 1
    with pytest.raises(ValueError):
        it.preview_ascii([], 1)


# --------------------------------------------------------------------------
# colour layers: one hue per Slack account
# --------------------------------------------------------------------------
#
# The invariants that matter here are the same one as above plus two more:
# ``max_segments`` is a budget across **all** layers together (every segment is
# one Chime message and the meeting shares ~150/s however many accounts draw),
# and a layer is only worth returning if it is worth a whole Slack account --
# so near-duplicate hues merge, greys collapse into one, and specks are dropped.

COLOR_MODES = ("auto", "outline", "threshold", "fill")
COLOR_BUDGETS = (3, 5, 9, 20, 60, 600)


@pytest.fixture
def flag(tmp_path: Path) -> Path:
    """Three equal vertical stripes, red / green / blue, filling the frame."""
    img = Image.new("RGB", (240, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 79, 159], fill=(220, 20, 20))
    draw.rectangle([80, 0, 159, 159], fill=(20, 180, 20))
    draw.rectangle([160, 0, 239, 159], fill=(20, 20, 220))
    return _save(img, tmp_path / "flag.png")


@pytest.fixture
def grey_patches(tmp_path: Path) -> Path:
    """Three grey tones on white: three clusters, not one hue between them."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 90, 190], fill=(30, 30, 30))
    draw.rectangle([100, 10, 150, 190], fill=(120, 120, 120))
    draw.rectangle([155, 10, 190, 190], fill=(190, 190, 190))
    return _save(img, tmp_path / "greys.png")


@pytest.fixture
def two_blues(tmp_path: Path) -> Path:
    """Two blues 7 degrees of hue apart: the same pastel once rendered."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 90, 180], fill=(0, 0, 200))  # hue 240
    draw.rectangle([110, 20, 180, 180], fill=(0, 30, 255))  # hue 233
    return _save(img, tmp_path / "blues.png")


@pytest.fixture
def speck(tmp_path: Path) -> Path:
    """Two big colour blocks plus a 16x16 red speck worth under 1% of the ink."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 120, 190], fill=(0, 0, 220))
    draw.rectangle([125, 10, 190, 190], fill=(0, 200, 0))
    draw.rectangle([2, 2, 17, 17], fill=(230, 0, 0))
    return _save(img, tmp_path / "speck.png")


@pytest.fixture
def dark_art(tmp_path: Path) -> Path:
    """Two colour blocks on a black page, for ``invert``."""
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 90, 180], fill=(255, 80, 80))
    draw.rectangle([110, 20, 180, 180], fill=(80, 255, 80))
    return _save(img, tmp_path / "dark.png")


@pytest.fixture
def grey_and_red(tmp_path: Path) -> Path:
    """A grey shape and a red shape, both well inside a white page."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 90, 180], fill=(110, 110, 110))
    draw.rectangle([110, 20, 180, 180], fill=(220, 20, 20))
    return _save(img, tmp_path / "greyred.png")


@pytest.fixture
def background_and_subject(tmp_path: Path) -> Path:
    """A flat lavender background plus two foreground shapes.

    The shape of the live failure: the background's only boundary is the edge
    of the image, so tracing it draws a box round the whole share.
    """
    img = Image.new("RGB", (240, 240), (150, 160, 215))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 110, 110], fill=(200, 40, 40))
    draw.rectangle([130, 130, 210, 210], fill=(40, 170, 60))
    return _save(img, tmp_path / "bg.png")


@pytest.fixture
def background_and_speckles(tmp_path: Path) -> Path:
    """One real subject plus 40 dots, each too small to be worth an attendee.

    The dots are big enough to survive ``min_contour_length`` on their own --
    the point is that a *layer* made of nothing but them is still noise.
    """
    img = Image.new("RGB", (240, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([60, 60, 180, 180], radius=12, fill=(40, 60, 190))
    for i in range(40):
        x, y = 4 + (i * 29) % 220, 4 + (i * 53) % 220
        draw.ellipse([x, y, x + 10, y + 10], fill=(210, 60, 20))
    return _save(img, tmp_path / "speckles.png")


@pytest.fixture
def crowded_pastels(tmp_path: Path) -> Path:
    """Three pastels whose hues (9, 26, 43) all land in one narrow arc."""
    img = Image.new("RGB", (240, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 100, 110], fill=(220, 160, 150))
    draw.rectangle([130, 10, 230, 110], fill=(220, 180, 150))
    draw.rectangle([10, 130, 110, 230], fill=(220, 200, 150))
    return _save(img, tmp_path / "pastels.png")


@pytest.fixture
def grey_with_a_blush(tmp_path: Path) -> Path:
    """A mostly-neutral shape with a saturated teal minority inside it."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], fill=(190, 190, 190))
    draw.rectangle([45, 45, 155, 155], fill=(0, 180, 180))
    return _save(img, tmp_path / "blush.png")


@pytest.fixture
def color_images(
    flag: Path,
    grey_patches: Path,
    two_blues: Path,
    speck: Path,
    dark_art: Path,
    square: Path,
    blank: Path,
    gradient: Path,
    transparent: Path,
) -> list[Path]:
    return [
        flag,
        grey_patches,
        two_blues,
        speck,
        dark_art,
        square,
        blank,
        gradient,
        transparent,
    ]


def layer_points(layer: it.ColorLayer) -> list[g.Point]:
    """Every coordinate of one layer that goes on the wire."""
    return [p for s in layer.segments for p in s.points]


def layer_bbox(layer: it.ColorLayer) -> g.Rect:
    """Bounding box of one layer's drawn curve (sampled, not control points)."""
    sampled = [
        g.bezier_point(s, i / 16) for s in layer.segments for i in range(17)
    ]
    return g.Rect.bounding(sampled)


def layers_inside(
    rect: g.Rect, layers: Sequence[it.ColorLayer], tol: float = 1e-9
) -> bool:
    r = rect.normalized()
    return all(
        r.x0 - tol <= p.x <= r.x1 + tol and r.y0 - tol <= p.y <= r.y1 + tol
        for layer in layers
        for p in layer_points(layer)
    )


# --------------------------------------------------------------------------
# hue arithmetic
# --------------------------------------------------------------------------


def test_rgb_to_hue_knows_the_primaries() -> None:
    assert it.rgb_to_hue((255, 0, 0)) == 0
    assert it.rgb_to_hue((255, 255, 0)) == 60
    assert it.rgb_to_hue((0, 255, 0)) == 120
    assert it.rgb_to_hue((0, 255, 255)) == 180
    assert it.rgb_to_hue((0, 0, 255)) == 240
    assert it.rgb_to_hue((255, 0, 255)) == 300


def test_rgb_to_hue_ignores_lightness_because_the_renderer_does() -> None:
    """Dark navy and pale sky blue are the same colour once drawn."""
    assert it.rgb_to_hue((0, 0, 60)) == it.rgb_to_hue((0, 0, 255)) == 240
    assert it.rgb_to_hue((170, 170, 255)) == 240
    # Greys have no hue at all.
    for grey in ((0, 0, 0), (128, 128, 128), (255, 255, 255)):
        assert it.rgb_to_hue(grey) == 0


def test_pastel_rgb_is_always_a_pastel() -> None:
    """``hsla(h, 100%, 75%)``: nothing dark, nothing grey, nothing white."""
    for hue in range(0, 360, 5):
        rendered = it.pastel_rgb(hue)
        assert min(rendered) == 128, rendered
        assert max(rendered) == 255, rendered
        assert it.rgb_to_hue(rendered) == pytest.approx(hue, abs=1)
    assert it.pastel_rgb(0) == (255, 128, 128)
    assert it.pastel_rgb(360) == it.pastel_rgb(0)


def test_hue_distance_wraps_around_the_circle() -> None:
    assert it._hue_distance(10, 350) == 20
    assert it._hue_distance(0, 180) == 180
    assert it._hue_distance(240, 240) == 0


# --------------------------------------------------------------------------
# the three-colour flag
# --------------------------------------------------------------------------


def test_flag_decomposes_into_three_sensible_hues(flag: Path) -> None:
    layers = it.trace_image_colors(flag, layers=3)
    assert len(layers) == 3, it.describe_layers(layers)
    for expected in (0, 120, 240):
        assert any(it._hue_distance(x.hue, expected) <= 20 for x in layers), (
            f"no layer near hue {expected}: {it.describe_layers(layers)}"
        )
    for layer in layers:
        assert layer.pixel_share == pytest.approx(1 / 3, abs=0.05)
        assert not layer.is_greyscale
        assert layer.hue == it.rgb_to_hue(layer.sample_rgb)
        assert layer.segments
        assert layer.trace is not None
    assert sum(x.pixel_share for x in layers) == pytest.approx(1.0, abs=0.02)


def test_flag_layers_are_cheap_and_report_their_own_cost(flag: Path) -> None:
    layers = it.trace_image_colors(flag, layers=3, max_segments=600)
    # Each stripe is one rectangle: a closed curve through four corners.
    for layer in layers:
        assert 4 <= len(layer) <= 10, layer.describe()
        assert layer.trace is not None
        assert layer.trace.contours == 1
        assert len(layer) == len(layer.trace.segments)
    total = sum(len(x) for x in layers)
    assert it.total_estimated_seconds(layers) == pytest.approx(total / 150.0)


def test_flag_layers_share_one_transform_so_the_colours_register(
    flag: Path,
) -> None:
    """Every layer must be fitted with the *same* scale, not its own.

    Fitted per layer, each stripe would be blown up to fill the rect and the
    three would land on top of each other instead of side by side.
    """
    rect = g.Rect(0.1, 0.2, 0.9, 0.8)
    layers = it.trace_image_colors(flag, rect, layers=3, max_segments=600)
    boxes = sorted((layer_bbox(x) for x in layers), key=lambda b: b.x0)
    assert len(boxes) == 3
    for box in boxes:
        # All three stripes are full height and a third of the width.
        assert box.y0 == pytest.approx(boxes[0].y0, abs=0.01)
        assert box.y1 == pytest.approx(boxes[0].y1, abs=0.01)
        assert box.width == pytest.approx(boxes[0].width, rel=0.1)
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left.x0 < right.x0, "the stripes must stay side by side"
        assert left.x1 < right.x1
    span = g.Rect.bounding(
        [g.Point(b.x0, b.y0) for b in boxes] + [g.Point(b.x1, b.y1) for b in boxes]
    )
    assert span.width > 0.6 * rect.width, "the layers together fill the rect"


def test_layers_is_an_upper_bound_not_a_promise(flag: Path) -> None:
    """A 3-colour image cannot use 8 accounts, however many you offer."""
    layers = it.trace_image_colors(flag, layers=8, max_segments=600)
    assert len(layers) == 3, it.describe_layers(layers)
    assert sorted(x.hue for x in layers) == sorted(
        y.hue for y in it.trace_image_colors(flag, layers=3, max_segments=600)
    )


# --------------------------------------------------------------------------
# the budget is shared by every layer
# --------------------------------------------------------------------------


def test_combined_segment_count_never_exceeds_the_budget(
    color_images: list[Path],
) -> None:
    for path in color_images:
        for mode in COLOR_MODES:
            for budget in COLOR_BUDGETS:
                layers = it.trace_image_colors(
                    path, layers=4, mode=mode, max_segments=budget
                )
                total = sum(len(x) for x in layers)
                assert total <= budget, (
                    f"{path.name}/{mode}/{budget}: {it.describe_layers(layers)}"
                )


def test_zero_budget_draws_nothing(flag: Path) -> None:
    assert it.trace_image_colors(flag, layers=3, max_segments=0) == []


def test_budget_is_split_in_proportion_to_ink(speck: Path) -> None:
    layers = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.0, max_segments=600
    )
    assert len(layers) == 3
    biggest, smallest = layers[0], layers[-1]
    assert biggest.pixel_share > smallest.pixel_share
    assert biggest.trace is not None and smallest.trace is not None
    assert biggest.trace.max_segments > smallest.trace.max_segments
    assert smallest.trace.max_segments >= it.MIN_LAYER_SEGMENTS


def test_a_tiny_budget_keeps_fewer_layers_rather_than_starving_them(
    flag: Path,
) -> None:
    """Better one account drawing a shape than three drawing two segments each."""
    counts = [
        len(it.trace_image_colors(flag, layers=3, max_segments=b))
        for b in (2, 5, 9, 60)
    ]
    assert counts == sorted(counts), counts
    assert counts[0] <= 1 < counts[-1]
    tight = it.trace_image_colors(flag, layers=3, max_segments=5)
    assert len(tight) == 1
    assert sum(len(x) for x in tight) <= 5
    assert tight[0].pixel_share == pytest.approx(1 / 3, abs=0.05), (
        "the share still describes the layer, not the budget"
    )


def test_unspent_budget_is_handed_to_the_layers_that_need_it(
    ring_and_specks: Path, tmp_path: Path
) -> None:
    """A simple layer cannot use its slice, so a complex one should get it."""
    img = Image.open(ring_and_specks).convert("RGB")
    pixels = img.load()
    assert pixels is not None
    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y][0] < 128:
                # The ring and the specks become a busy green layer...
                pixels[x, y] = (0, 160, 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([120, 120, 180, 180], fill=(200, 0, 0))  # ...this one is 1 box
    path = _save(img, tmp_path / "mixed.png")

    layers = it.trace_image_colors(
        path, layers=2, blur=0.0, min_contour_length=0.0, max_segments=120
    )
    assert len(layers) == 2
    busy = max(layers, key=lambda x: len(x))
    simple = min(layers, key=lambda x: len(x))
    assert busy.trace is not None and simple.trace is not None
    assert len(simple) < simple.trace.max_segments, "it could not spend its slice"
    assert busy.trace.max_segments > 120 * busy.pixel_share + 1, (
        "the busy layer should have been given the slack"
    )
    assert sum(len(x) for x in layers) <= 120


def test_layers_degrade_like_trace_image_does(flag: Path) -> None:
    generous = it.trace_image_colors(flag, layers=3, max_segments=600)
    tight = it.trace_image_colors(flag, layers=3, max_segments=15)
    assert sum(len(x) for x in tight) <= 15
    for layer in tight:
        assert layer.trace is not None
        assert layer.trace.simplify_epsilon >= generous[0].trace.simplify_epsilon


# --------------------------------------------------------------------------
# ordering, shares and filtering
# --------------------------------------------------------------------------


def test_layers_are_sorted_by_ink_share_largest_first(speck: Path) -> None:
    layers = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.0, max_segments=600
    )
    shares = [x.pixel_share for x in layers]
    assert shares == sorted(shares, reverse=True), shares
    assert len(set(shares)) == 3, "this image has three distinctly sized layers"


def test_shares_sum_to_one_before_filtering(speck: Path) -> None:
    layers = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.0, max_segments=600
    )
    assert sum(x.pixel_share for x in layers) == pytest.approx(1.0, abs=1e-6)


def test_min_layer_share_drops_a_speck(speck: Path) -> None:
    keep_all = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.0, max_segments=600
    )
    filtered = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.02, max_segments=600
    )
    assert len(keep_all) == 3
    assert keep_all[-1].pixel_share < 0.02
    assert it._hue_distance(keep_all[-1].hue, 0) <= 20, "the speck is red"
    assert len(filtered) == 2, it.describe_layers(filtered)
    assert all(x.pixel_share >= 0.02 for x in filtered)
    # Shares are never renormalized, so the shortfall is visible.
    assert sum(x.pixel_share for x in filtered) < 1.0


def test_min_layer_share_can_drop_everything(flag: Path) -> None:
    assert it.trace_image_colors(flag, layers=3, min_layer_share=1.0) == []


# --------------------------------------------------------------------------
# hues that render identically are merged
# --------------------------------------------------------------------------


def test_two_near_identical_blues_become_one_layer(two_blues: Path) -> None:
    """7 degrees apart is the same pastel: one account, not two."""
    layers = it.trace_image_colors(
        two_blues, layers=3, blur=0.0, max_segments=600
    )
    assert len(layers) == 1, it.describe_layers(layers)
    layer = layers[0]
    assert it._hue_distance(layer.hue, 240) <= it.DEFAULT_HUE_MERGE_DEGREES
    assert layer.merged_clusters == 2
    assert layer.pixel_share == pytest.approx(1.0, abs=1e-6)
    assert layer.trace is not None
    assert layer.trace.contours == 2, "both blue blocks are in the one layer"


def test_merging_can_be_switched_off(two_blues: Path) -> None:
    layers = it.trace_image_colors(
        two_blues,
        layers=3,
        blur=0.0,
        max_segments=600,
        hue_merge_degrees=0.0,
        spread_hues=False,  # tested on its own below; it would pull them apart
    )
    assert len(layers) == 2
    assert all(x.merged_clusters == 1 for x in layers)
    gap = it._hue_distance(layers[0].hue, layers[1].hue)
    assert 0 < gap <= it.DEFAULT_HUE_MERGE_DEGREES, (
        "two hues this close are what the default merges"
    )


def test_distinct_hues_are_not_merged(flag: Path) -> None:
    layers = it.trace_image_colors(
        flag, layers=3, max_segments=600, hue_merge_degrees=30.0
    )
    assert len(layers) == 3
    assert all(x.merged_clusters == 1 for x in layers)


# --------------------------------------------------------------------------
# greyscale has no hue
# --------------------------------------------------------------------------


def test_greyscale_image_gives_one_layer_not_several_meaningless_hues(
    grey_patches: Path,
) -> None:
    layers = it.trace_image_colors(grey_patches, layers=4, max_segments=600)
    assert len(layers) == 1, it.describe_layers(layers)
    layer = layers[0]
    assert layer.is_greyscale
    assert layer.hue == it.DEFAULT_GREY_HUE, "documented as arbitrary"
    assert layer.merged_clusters >= 2
    assert layer.pixel_share == pytest.approx(1.0, abs=1e-6)
    assert layer.trace is not None
    assert layer.trace.contours >= 3, "all three patches are in the one layer"
    assert "near-neutral" in layer.describe()


def test_a_greyscale_gradient_is_one_layer_too(gradient: Path) -> None:
    layers = it.trace_image_colors(gradient, layers=4, max_segments=600)
    assert len(layers) <= 1
    assert all(x.is_greyscale for x in layers)


def test_a_black_and_white_drawing_is_one_greyscale_layer(square: Path) -> None:
    layers = it.trace_image_colors(square, layers=3, max_segments=600)
    assert len(layers) == 1
    assert layers[0].is_greyscale


def test_the_neutral_hue_moves_out_of_a_real_colour_s_way(
    grey_and_red: Path,
) -> None:
    """A neutral's hue is arbitrary, so it must not duplicate a real colour."""
    layers = it.trace_image_colors(grey_and_red, layers=3, max_segments=600)
    grey = [x for x in layers if x.is_greyscale]
    colored = [x for x in layers if not x.is_greyscale]
    assert grey and colored, it.describe_layers(layers)
    assert any(it._hue_distance(x.hue, 0) <= 20 for x in colored), "red is hue 0"
    assert grey[0].hue != it.DEFAULT_GREY_HUE, "0 was taken by the red layer"
    for layer in colored:
        assert (
            it._hue_distance(grey[0].hue, layer.hue)
            >= it.DEFAULT_MIN_HUE_SEPARATION
        )


def test_grey_hue_is_configurable(grey_patches: Path) -> None:
    layers = it.trace_image_colors(
        grey_patches, layers=3, grey_hue=200, max_segments=600
    )
    assert [x.hue for x in layers] == [200]


def test_near_white_but_chromatic_regions_are_still_ink(tmp_path: Path) -> None:
    """Pale yellow is light, but it has a hue, so it is drawable."""
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([30, 30, 129, 129], fill=(255, 250, 190))
    path = _save(img, tmp_path / "pale.png")
    layers = it.trace_image_colors(path, layers=2, max_segments=600)
    assert len(layers) == 1
    assert it._hue_distance(layers[0].hue, 55) <= 20, layers[0].describe()
    assert not layers[0].is_greyscale


# --------------------------------------------------------------------------
# nothing to draw
# --------------------------------------------------------------------------


def test_blank_image_produces_no_layers(blank: Path) -> None:
    assert it.trace_image_colors(blank, layers=3) == []
    assert it.total_estimated_seconds([]) == 0.0


def test_uniform_colour_images_produce_no_layers(tmp_path: Path) -> None:
    for colour in ((255, 255, 255), (0, 0, 0), (200, 30, 30)):
        path = _save(
            Image.new("RGB", (64, 64), colour), tmp_path / f"flat{colour[0]}.png"
        )
        assert it.trace_image_colors(path, layers=3) == [], colour


def test_an_all_page_image_produces_no_layers(square: Path) -> None:
    """With everything below the background line there is no ink left."""
    assert it.trace_image_colors(square, layers=3, background_level=0.0) == []


def test_one_pixel_image_does_not_crash_in_colour(one_pixel: Path) -> None:
    assert it.trace_image_colors(one_pixel, layers=3) == []


# --------------------------------------------------------------------------
# backgrounds and invert
# --------------------------------------------------------------------------


def test_invert_treats_the_dark_page_as_background(dark_art: Path) -> None:
    """A black page is ink unless ``invert`` says otherwise -- or it is dropped.

    Two independent defences catch it: ``invert=True`` makes near-black *page*
    (so it is never ink), and ``drop_background`` discards whichever cluster
    owns the image border. Turning both off is what the live artifact looked
    like: the page becomes the biggest layer in the picture.
    """
    kept = it.trace_image_colors(
        dark_art, layers=3, max_segments=600, drop_background=False
    )
    assert kept[0].is_greyscale, "the page is the biggest 'colour'"
    assert kept[0].is_background
    assert kept[0].pixel_share > 0.4

    for layers in (
        it.trace_image_colors(dark_art, layers=3, max_segments=600),
        it.trace_image_colors(
            dark_art, layers=3, max_segments=600, invert=True
        ),
    ):
        colored = [x for x in layers if not x.is_greyscale]
        assert len(colored) == 2, it.describe_layers(layers)
        assert not any(x.is_background for x in layers)
        for expected in (0, 120):
            assert any(
                it._hue_distance(x.hue, expected) <= 20 for x in colored
            ), it.describe_layers(layers)


def test_transparency_is_composited_onto_white_in_colour(
    transparent: Path,
) -> None:
    """Transparent must read as page, not as a big black layer."""
    layers = it.trace_image_colors(transparent, layers=3, max_segments=600)
    assert len(layers) == 1
    assert layers[0].is_greyscale
    box = layer_bbox(layers[0])
    assert 0.5 < box.width < 0.95, "the opaque square, not the whole frame"


# --------------------------------------------------------------------------
# one layer is the single-layer trace
# --------------------------------------------------------------------------


def test_one_layer_behaves_like_the_single_layer_trace(square: Path) -> None:
    single = it.trace_image(square, mode="threshold", max_segments=600)
    layers = it.trace_image_colors(square, layers=1, max_segments=600)
    assert len(layers) == 1
    layer = layers[0]
    assert layer.pixel_share == pytest.approx(1.0)
    assert layer.merged_clusters == 1
    assert abs(len(layer) - len(single)) <= 2, (
        f"{layer.describe()} vs {single.describe()}"
    )
    box, reference = layer_bbox(layer), bbox(single)
    assert box.width == pytest.approx(reference.width, rel=0.1)
    assert box.height == pytest.approx(reference.height, rel=0.1)
    assert layer.estimated_seconds() == pytest.approx(len(layer) / 150.0)


def test_trace_image_is_unchanged_by_a_colour_trace(square: Path) -> None:
    """The colour tracer is additive: it must not perturb the old path."""
    before = it.trace_image(square, mode="threshold")
    it.trace_image_colors(square, layers=3)
    after = it.trace_image(square, mode="threshold")
    assert before.segments == after.segments
    assert before.describe() == after.describe()


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_every_coordinate_is_inside_the_requested_rect(
    flag: Path, speck: Path
) -> None:
    rects = (
        g.Rect(0.0, 0.0, 1.0, 1.0),
        g.Rect(0.6, 0.1, 0.95, 0.45),
        g.Rect(0.0, 0.0, 0.1, 1.0),
        g.Rect(0.9, 0.9, 1.0, 1.0),
        g.Rect(0.8, 0.2, 0.2, 0.8),  # deliberately reversed
    )
    for path in (flag, speck):
        for rect in rects:
            for mode in COLOR_MODES:
                layers = it.trace_image_colors(
                    path, rect, layers=3, mode=mode, max_segments=300
                )
                assert layers, (path.name, rect, mode)
                assert layers_inside(rect, layers), (path.name, rect, mode)
                assert layers_inside(g.unit_square(), layers)
                for layer in layers:
                    assert layer.trace is not None
                    assert layer.trace.rect == rect.normalized()


def test_everything_is_normalized_without_a_rect(
    color_images: list[Path],
) -> None:
    unit = g.unit_square()
    for path in color_images:
        for mode in COLOR_MODES:
            layers = it.trace_image_colors(
                path, layers=3, mode=mode, max_segments=300
            )
            assert layers_inside(unit, layers), f"{path.name}/{mode} left 0..1"


def _span(layers: Sequence[it.ColorLayer]) -> g.Rect:
    """Bounding box of every layer's drawn curve together."""
    boxes = [layer_bbox(x) for x in layers]
    return g.Rect.bounding(
        [g.Point(b.x0, b.y0) for b in boxes] + [g.Point(b.x1, b.y1) for b in boxes]
    )


def test_aspect_can_be_stretched_on_request_across_layers(flag: Path) -> None:
    """``preserve_aspect`` applies to the decomposition as a whole.

    The absolute aspect of the span is not the image's: Catmull-Rom bulges a
    rectangle's edges outward (a traced square is a squarish blob), and a
    stripe is tall and thin. What must hold is that stretching fills the rect's
    height and that both axes move together for every layer.
    """
    preserved = _span(it.trace_image_colors(flag, layers=3, max_segments=600))
    stretched = _span(
        it.trace_image_colors(
            flag, layers=3, max_segments=600, preserve_aspect=False
        )
    )
    assert stretched.height > preserved.height * 1.2
    assert stretched.width == pytest.approx(preserved.width, rel=0.01)
    assert stretched.width / stretched.height < preserved.width / preserved.height


# --------------------------------------------------------------------------
# fill mode
# --------------------------------------------------------------------------


def test_fill_mode_fills_each_colour_with_hairpins(flag: Path) -> None:
    outlined = it.trace_image_colors(
        flag, layers=3, mode="threshold", max_segments=600
    )
    filled = it.trace_image_colors(flag, layers=3, mode="fill", max_segments=600)
    assert len(filled) == 3
    assert sum(len(x) for x in filled) > sum(len(x) for x in outlined) * 2
    for layer in filled:
        assert layer.trace is not None
        assert layer.trace.mode == "fill"
        assert layer.trace.fill_row_height is not None
        for segment in layer.segments:
            assert segment.a.y == pytest.approx(segment.b.y)
            assert segment.c.y == pytest.approx(segment.d.y)
            assert segment.d.y > segment.a.y
            assert segment.a.x == pytest.approx(segment.d.x)


def test_fill_layers_stay_in_their_own_stripe(flag: Path) -> None:
    layers = it.trace_image_colors(flag, layers=3, mode="fill", max_segments=600)
    boxes = sorted((layer_bbox(x) for x in layers), key=lambda b: b.x0)
    for left, right in zip(boxes, boxes[1:], strict=False):
        assert left.x1 <= right.x1
        assert left.x0 < right.x0


# --------------------------------------------------------------------------
# ColorLayer
# --------------------------------------------------------------------------


def _layer(n: int, hue: int = 200) -> it.ColorLayer:
    segment = g.Segment.line((0.0, 0.0), (0.1, 0.1))
    return it.ColorLayer(
        hue=hue, segments=[segment] * n, pixel_share=0.5, sample_rgb=(0, 0, 200)
    )


def test_layer_estimated_seconds_matches_trace_result() -> None:
    assert _layer(0).estimated_seconds() == 0.0
    assert _layer(150).estimated_seconds() == pytest.approx(1.0)
    assert _layer(2000).estimated_seconds() == pytest.approx(13.33, rel=1e-3)
    assert _layer(75).estimated_seconds(rate=75.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _layer(10).estimated_seconds(rate=0.0)


def test_total_estimated_seconds_adds_up_because_the_bucket_is_shared() -> None:
    """Six accounts drawing at once drain one ~150/s bucket six times as fast."""
    layers = [_layer(50), _layer(50), _layer(50)]
    assert it.total_estimated_seconds(layers) == pytest.approx(1.0)
    assert it.total_estimated_seconds(layers) == pytest.approx(
        sum(x.estimated_seconds() for x in layers)
    )
    assert it.total_estimated_seconds(layers, rate=75.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        it.total_estimated_seconds(layers, rate=0.0)


def test_layer_len_bool_and_colours() -> None:
    layer = _layer(3, hue=240)
    assert len(layer) == 3
    assert layer
    assert not _layer(0)
    assert layer.css_color() == "hsla(240, 100%, 75%, 1)"
    assert layer.css_color(0.5) == "hsla(240, 100%, 75%, 0.5)"
    assert layer.rendered_rgb == it.pastel_rgb(240) == (128, 128, 255)
    assert layer.sample_rgb == (0, 0, 200), "the colour it came from is kept"


def test_describe_layers_reports_accounts_and_cost(flag: Path) -> None:
    layers = it.trace_image_colors(flag, layers=3)
    text = it.describe_layers(layers)
    assert "3 layer(s)" in text, text
    assert f"{sum(len(x) for x in layers)} segments" in text
    assert "s at 150/s" in text
    assert text.count("hue=") == 3
    assert "% of ink" in text


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def test_bad_arguments_raise(flag: Path) -> None:
    for kwargs in (
        {"layers": 0},
        {"layers": -1},
        {"mode": "rotoscope"},
        {"max_segments": -1},
        {"simplify": -0.1},
        {"min_contour_length": -1.0},
        {"max_dimension": 0},
        {"min_layer_share": 1.5},
        {"min_layer_share": -0.1},
        {"hue_merge_degrees": -1.0},
        {"min_chroma": 2.0},
        {"background_level": -0.5},
        {"min_layer_segments": 0},
    ):
        with pytest.raises(ValueError):
            it.trace_image_colors(flag, **kwargs)  # type: ignore[arg-type]


def test_edges_is_an_alias_in_colour_mode_too(flag: Path) -> None:
    assert [
        (x.hue, len(x))
        for x in it.trace_image_colors(flag, layers=3, mode="edges")
    ] == [
        (x.hue, len(x))
        for x in it.trace_image_colors(flag, layers=3, mode="outline")
    ]


def test_accepts_an_open_pil_image(flag: Path) -> None:
    with Image.open(flag) as img:
        layers = it.trace_image_colors(img, layers=3)
    assert len(layers) == 3


def test_large_colour_images_are_downscaled(tmp_path: Path) -> None:
    img = Image.new("RGB", (1600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 700, 800], fill=(200, 0, 0))
    draw.ellipse([800, 100, 1500, 800], fill=(0, 0, 200))
    path = _save(img, tmp_path / "big.png")
    layers = it.trace_image_colors(path, layers=2, max_segments=600)
    assert len(layers) == 2
    for layer in layers:
        assert layer.trace is not None
        assert layer.trace.source_size == (1600, 900)
        assert max(layer.trace.traced_size) == it.DEFAULT_MAX_DIMENSION
    assert sum(len(x) for x in layers) < 200


def test_deterministic(flag: Path) -> None:
    """Same image, same layers: the quantiser must not be randomly seeded."""
    first = it.trace_image_colors(flag, layers=4, max_segments=300)
    second = it.trace_image_colors(flag, layers=4, max_segments=300)
    assert [(x.hue, x.segments, x.pixel_share) for x in first] == [
        (y.hue, y.segments, y.pixel_share) for y in second
    ]


# --------------------------------------------------------------------------
# the background is not the picture
# --------------------------------------------------------------------------


def test_the_background_layer_is_dropped(background_and_subject: Path) -> None:
    """The lavender page must not become a layer, let alone the biggest one."""
    layers = it.trace_image_colors(
        background_and_subject, layers=3, max_segments=400
    )
    assert len(layers) == 2, it.describe_layers(layers)
    for expected in (0, 120):  # the red circle and the green square
        assert any(it._hue_distance(x.hue, expected) <= 25 for x in layers), (
            it.describe_layers(layers)
        )
    assert not any(x.is_background for x in layers)
    # Nothing left spans the whole frame.
    for layer in layers:
        box = layer_bbox(layer)
        assert box.width < 0.8 or box.height < 0.8, layer.describe()


def test_dropping_the_background_hands_its_budget_to_the_subject(
    background_and_subject: Path,
) -> None:
    budget = 400
    dropped = it.trace_image_colors(
        background_and_subject, layers=3, max_segments=budget
    )
    kept = it.trace_image_colors(
        background_and_subject,
        layers=3,
        max_segments=budget,
        drop_background=False,
    )
    background = [x for x in kept if x.is_background]
    assert len(background) == 1, it.describe_layers(kept)
    assert background[0].pixel_share > 0.5, "the page is most of the image"
    # The two subjects are budgeted against the whole of max_segments once the
    # background is out of the running, instead of their share of it.
    for layer in dropped:
        assert layer.trace is not None
        twin = next(
            x for x in kept if it._hue_distance(x.hue, layer.hue) <= 25
        )
        assert twin.trace is not None
        assert layer.trace.max_segments > twin.trace.max_segments
    assert sum(len(x) for x in dropped) <= budget
    assert sum(len(x) for x in kept) <= budget


def test_keeping_the_background_reproduces_the_old_behaviour(
    background_and_subject: Path,
) -> None:
    kept = it.trace_image_colors(
        background_and_subject,
        layers=3,
        max_segments=400,
        drop_background=False,
        frame_coverage=1.0,
    )
    assert len(kept) == 3
    background = next(x for x in kept if x.is_background)
    assert background.trace is not None
    assert background.trace.frames_dropped == 0
    assert "BACKGROUND" in background.describe()
    # ...and this is what it looked like: a box around the whole frame.
    box = layer_bbox(background)
    assert box.width > 0.85 and box.height > 0.85


def test_the_frame_contour_is_gone_even_when_the_background_is_kept(
    background_and_subject: Path,
) -> None:
    """Defence in depth: the frame goes even if the layer stays."""
    kept = it.trace_image_colors(
        background_and_subject,
        layers=3,
        max_segments=400,
        drop_background=False,
    )
    background = next(x for x in kept if x.is_background)
    assert background.trace is not None
    assert background.trace.frames_dropped >= 1
    for layer in kept:
        box = layer_bbox(layer)
        assert box.width < 0.9 or box.height < 0.9, layer.describe()


def test_trace_image_does_not_draw_the_frame_of_such_an_image(
    background_and_subject: Path,
) -> None:
    """Single-colour tracing has the same artifact, so it gets the same fix."""
    for mode in ("threshold", "outline", "auto"):
        result = it.trace_image(
            background_and_subject, mode=mode, max_segments=400
        )
        sampled = [
            g.bezier_point(s, i / 16) for s in result.segments for i in range(17)
        ]
        if not sampled:
            continue
        box = g.Rect.bounding(sampled)
        assert box.width < 0.95 or box.height < 0.95, (
            f"{mode} drew a box around the whole rect: {result.describe()}"
        )


def test_an_image_where_every_cluster_touches_the_border_keeps_them_all(
    flag: Path,
) -> None:
    """Three stripes all reach the edge: there is no background to single out."""
    layers = it.trace_image_colors(flag, layers=3, max_segments=400)
    assert len(layers) == 3, it.describe_layers(layers)
    assert not any(x.is_background for x in layers)


def test_a_logo_on_white_has_no_background_cluster(speck: Path) -> None:
    """The border is page, not ink, so nothing is dropped as background."""
    layers = it.trace_image_colors(
        speck, layers=3, blur=0.0, min_layer_share=0.0, max_segments=600
    )
    assert len(layers) == 3
    assert sum(x.pixel_share for x in layers) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# speckle is not worth an attendee
# --------------------------------------------------------------------------


def test_a_layer_of_nothing_but_dots_is_dropped(
    background_and_speckles: Path,
) -> None:
    layers = it.trace_image_colors(
        background_and_speckles, layers=3, max_segments=400, blur=0.0
    )
    assert len(layers) == 1, it.describe_layers(layers)
    assert it._hue_distance(layers[0].hue, 230) <= 30, "the blue subject"
    assert layers[0].trace is not None
    box = layer_bbox(layers[0])
    assert box.width > 0.4 and box.height > 0.4, "the subject, not the dots"

    kept = it.trace_image_colors(
        background_and_speckles,
        layers=3,
        max_segments=400,
        blur=0.0,
        min_layer_feature=0.0,
    )
    assert len(kept) == 2, it.describe_layers(kept)
    speckles = next(x for x in kept if it._hue_distance(x.hue, 230) > 30)
    assert speckles.trace is not None
    assert speckles.trace.contours > 10, "dozens of dots"
    # ...and each one is tiny, which is exactly the rule that drops it.
    assert speckles.trace.contours_found >= 40


def test_speckle_only_images_keep_their_best_layer(tmp_path: Path) -> None:
    """Dropping every layer would be worse than drawing the dots."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(30):
        x, y = 5 + (i * 31) % 190, 5 + (i * 47) % 190
        draw.ellipse([x, y, x + 4, y + 4], fill=(200, 0, 0))
    path = _save(img, tmp_path / "dots.png")
    layers = it.trace_image_colors(
        path, layers=3, max_segments=400, min_contour_length=0.0
    )
    assert len(layers) == 1
    assert layers[0].segments


def test_the_colour_default_filters_more_noise_than_trace_image_does() -> None:
    assert it.DEFAULT_MIN_LAYER_CONTOUR > it.DEFAULT_MIN_CONTOUR_LENGTH


# --------------------------------------------------------------------------
# hues a viewer can tell apart
# --------------------------------------------------------------------------


def test_crowded_pastel_hues_are_spread_apart(crowded_pastels: Path) -> None:
    measured = it.trace_image_colors(
        crowded_pastels, layers=3, max_segments=600, spread_hues=False
    )
    spread = it.trace_image_colors(crowded_pastels, layers=3, max_segments=600)
    assert len(measured) == len(spread) == 3, it.describe_layers(spread)

    # As measured, all three sit in one narrow arc and would render as three
    # near-identical pastels.
    hues = [x.hue for x in measured]
    assert max(
        it._hue_distance(a, b) for a in hues for b in hues
    ) < it.DEFAULT_MIN_HUE_SEPARATION
    assert all(x.original_hue is None for x in measured)

    # Spread, every pair is distinguishable.
    for a, b in itertools.combinations(spread, 2):
        assert it._hue_distance(a.hue, b.hue) >= it.DEFAULT_MIN_HUE_SEPARATION - 1, (
            it.describe_layers(spread)
        )
    moved = [x for x in spread if x.original_hue is not None]
    assert len(moved) == 2, "the largest layer keeps its hue; the others move"
    assert spread[0].original_hue is None


def test_spreading_preserves_the_order_around_the_wheel(
    crowded_pastels: Path,
) -> None:
    measured = it.trace_image_colors(
        crowded_pastels, layers=3, max_segments=600, spread_hues=False
    )
    spread = it.trace_image_colors(crowded_pastels, layers=3, max_segments=600)
    before = {x.sample_rgb: x.hue for x in measured}
    after = {x.sample_rgb: x.hue for x in spread}
    assert set(before) == set(after)

    def cyclic(order: dict[tuple[int, int, int], int]) -> list[tuple[int, ...]]:
        """The colours in hue order, rotated so the first colour leads."""
        ranked = sorted(order, key=lambda rgb: order[rgb])
        pivot = ranked.index(sorted(order)[0])
        return [tuple(c) for c in ranked[pivot:] + ranked[:pivot]]

    assert cyclic(before) == cyclic(after), (
        f"relative order changed: {before} -> {after}"
    )


def test_vivid_hues_are_left_where_they_are(flag: Path) -> None:
    """Red/green/blue are already 120 apart: spreading must not touch them."""
    spread = it.trace_image_colors(flag, layers=3, max_segments=600)
    measured = it.trace_image_colors(
        flag, layers=3, max_segments=600, spread_hues=False
    )
    assert [x.hue for x in spread] == [x.hue for x in measured]
    assert all(x.original_hue is None for x in spread)
    for expected in (0, 120, 240):
        assert any(it._hue_distance(x.hue, expected) <= 10 for x in spread)


def test_spreading_can_be_switched_off(crowded_pastels: Path) -> None:
    layers = it.trace_image_colors(
        crowded_pastels, layers=3, max_segments=600, spread_hues=False
    )
    assert all(x.original_hue is None for x in layers)
    assert all(x.hue < 60 for x in layers), "the measured, crowded hues"


def test_a_saturated_minority_decides_the_layer_s_hue(
    grey_with_a_blush: Path,
) -> None:
    """Chroma-weighted, so the pixels that have colour choose it.

    Three quarters of this layer is neutral grey, whose nominal hue is 0. An
    unweighted average would therefore come out red; weighting by chroma gives
    the teal minority the vote it deserves.
    """
    layers = it.trace_image_colors(
        grey_with_a_blush, layers=1, max_segments=600
    )
    assert len(layers) == 1
    layer = layers[0]
    assert not layer.is_greyscale, layer.describe()
    assert it._hue_distance(layer.hue, 180) <= 15, layer.describe()
    assert it._hue_distance(layer.hue, 0) > 60
    assert 0.15 < layer.chroma < 0.4
    assert it.rgb_to_hue(layer.sample_rgb) == pytest.approx(180, abs=15), (
        "the representative colour is the saturated one, not the average"
    )


def test_a_cluster_whose_hues_disagree_gets_no_confident_colour(
    tmp_path: Path,
) -> None:
    """Red plus green averages to a confident yellow that is not in the image."""
    img = Image.new("RGB", (120, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 55, 110], fill=(220, 20, 20))
    draw.rectangle([65, 10, 110, 110], fill=(20, 220, 20))
    path = _save(img, tmp_path / "redgreen.png")
    forced = it.trace_image_colors(path, layers=1, max_segments=600)
    assert len(forced) == 1
    assert forced[0].is_greyscale, forced[0].describe()
    assert it._hue_distance(forced[0].hue, 60) > 30, "not yellow"
    # With two layers to work with, both colours come back properly.
    both = it.trace_image_colors(path, layers=2, max_segments=600)
    assert len(both) == 2
    assert not any(x.is_greyscale for x in both)


def test_an_all_neutral_image_does_not_produce_vivid_layers(
    grey_patches: Path, tmp_path: Path
) -> None:
    for path in (grey_patches, _save(
        Image.linear_gradient("L").resize((160, 160)), tmp_path / "ramp.png"
    )):
        layers = it.trace_image_colors(path, layers=4, max_segments=600)
        assert len(layers) <= 1, it.describe_layers(layers)
        assert all(x.is_greyscale for x in layers)
        assert all(x.chroma < it.DEFAULT_MIN_CHROMA for x in layers)


def test_a_pale_wash_is_treated_as_neutral(tmp_path: Path) -> None:
    """0.16 chroma renders as a vivid colour, so it does not get one."""
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([20, 20, 139, 139], fill=(239, 201, 198))
    path = _save(img, tmp_path / "cream.png")
    layers = it.trace_image_colors(path, layers=2, max_segments=600)
    assert len(layers) == 1
    assert layers[0].chroma < it.DEFAULT_MIN_CHROMA
    assert layers[0].is_greyscale, layers[0].describe()
    assert layers[0].hue == it.DEFAULT_GREY_HUE


def test_determinism_including_the_hue_choice(
    crowded_pastels: Path, background_and_subject: Path
) -> None:
    for path in (crowded_pastels, background_and_subject):
        runs = [
            [
                (x.hue, x.original_hue, x.sample_rgb, len(x))
                for x in it.trace_image_colors(path, layers=4, max_segments=300)
            ]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2], path.name


def test_describe_shows_the_source_colour_and_the_drawn_colour(
    crowded_pastels: Path,
) -> None:
    layers = it.trace_image_colors(crowded_pastels, layers=3, max_segments=600)
    text = it.describe_layers(layers)
    for layer in layers:
        assert "rgb({},{},{}) -> rgb({},{},{})".format(
            *layer.sample_rgb, *layer.rendered_rgb
        ) in text
        assert layer.sample_rgb != layer.rendered_rgb, (
            "a pastel source still renders as something else entirely"
        )
    assert "chroma=" in text
    assert "hue moved from" in text


# --------------------------------------------------------------------------
# the perceptual colour space
# --------------------------------------------------------------------------


def test_oklab_matches_the_reference_values() -> None:
    """Spot values from Ottosson's definition, to a few decimals."""
    import numpy as np

    white = it._rgb_to_oklab(np.array([255.0, 255.0, 255.0]))
    assert white[0] == pytest.approx(1.0, abs=2e-3)
    assert white[1] == pytest.approx(0.0, abs=2e-3)
    assert white[2] == pytest.approx(0.0, abs=2e-3)
    black = it._rgb_to_oklab(np.array([0.0, 0.0, 0.0]))
    assert list(black) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    red = it._rgb_to_oklab(np.array([255.0, 0.0, 0.0]))
    assert red[0] == pytest.approx(0.628, abs=5e-3)
    assert red[1] == pytest.approx(0.225, abs=5e-3)
    assert red[2] == pytest.approx(0.126, abs=5e-3)


def test_oklab_spaces_lightness_more_evenly_than_rgb() -> None:
    """Why cluster there: equal RGB steps are not equal perceptual steps."""
    import numpy as np

    greys = it._rgb_to_oklab(
        np.array([[0.0] * 3, [64.0] * 3, [128.0] * 3, [192.0] * 3, [255.0] * 3])
    )[:, 0]
    steps = np.diff(greys)
    assert steps.min() > 0.0
    # In RGB the four steps are identical; in OkLab the dark end is stretched,
    # which is what stops k-means lumping every shadow together.
    assert steps.max() / steps.min() > 1.5


def test_chroma_and_weighted_hue() -> None:
    import numpy as np

    assert it._rgb_chroma(np.array([255.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert it._rgb_chroma(np.array([128.0] * 3)) == pytest.approx(0.0)
    # Circular: the mean of 350 and 10 is 0, not 180.
    hue, concentration = it._weighted_hue(
        np.array([350.0, 10.0]), np.array([1.0, 1.0])
    )
    assert hue == 0
    assert concentration > 0.98
    # Weighted: the heavier hue wins.
    hue, _ = it._weighted_hue(np.array([0.0, 120.0]), np.array([1.0, 9.0]))
    assert it._hue_distance(hue, 120) < 15
    # Opposed hues cancel: no usable direction.
    _, concentration = it._weighted_hue(
        np.array([0.0, 120.0, 240.0]), np.array([1.0, 1.0, 1.0])
    )
    assert concentration < 1e-9
    assert it._weighted_hue(np.array([]), np.array([])) == (0, 0.0)


def test_spread_hues_is_a_no_op_when_there_is_nothing_to_fix() -> None:
    assert it._spread_hues([10], 0, 40.0) == [10]
    assert it._spread_hues([0, 120, 240], 0, 40.0) == [0, 120, 240]
    assert it._spread_hues([0, 120, 240], 0, 0.0) == [0, 120, 240]


def test_spread_hues_separates_and_keeps_the_anchor() -> None:
    out = it._spread_hues([10, 20, 30], 1, 40.0)
    assert out[1] == 20, "the anchor does not move"
    assert len(set(out)) == 3
    for a, b in itertools.combinations(out, 2):
        assert it._hue_distance(a, b) >= 39


def test_spread_hues_spaces_evenly_when_the_wheel_is_too_small() -> None:
    out = it._spread_hues(list(range(10)), 0, 40.0)  # 10 * 40 > 360
    assert len(set(out)) == 10
    gaps = sorted(
        it._hue_distance(a, b) for a, b in itertools.combinations(out, 2)
    )
    assert gaps[0] == pytest.approx(36, abs=1)
