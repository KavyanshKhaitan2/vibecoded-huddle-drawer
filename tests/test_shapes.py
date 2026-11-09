"""Tests for the named shapes and the SVG path subset."""

from __future__ import annotations

import math

import pytest

from huddle_drawings.geometry import Rect, bezier_point
from huddle_drawings.shapes import (
    SHAPES,
    shape_segments,
    svg_path_segments,
    text_in,
)

RECT = Rect(0.25, 0.25, 0.75, 0.75)
EPS = 1e-6


def _all_points(segments):
    for s in segments:
        yield from (s.a, s.b, s.c, s.d)


@pytest.mark.parametrize("name", SHAPES)
def test_shapes_are_non_empty_and_fit_the_rect(name: str) -> None:
    segs = shape_segments(name, RECT)
    assert segs, f"{name} produced no segments"
    xs = [p.x for p in _all_points(segs)]
    ys = [p.y for p in _all_points(segs)]
    assert min(xs) >= RECT.x0 - EPS
    assert max(xs) <= RECT.x1 + EPS
    assert min(ys) >= RECT.y0 - EPS
    assert max(ys) <= RECT.y1 + EPS


@pytest.mark.parametrize("name", SHAPES)
def test_shapes_default_to_the_whole_frame(name: str) -> None:
    for p in _all_points(shape_segments(name)):
        assert -EPS <= p.x <= 1 + EPS
        assert -EPS <= p.y <= 1 + EPS


def test_unknown_shape_lists_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown shape"):
        shape_segments("dodecahedron")


def test_curved_shapes_are_cheaper_than_their_sampled_polyline() -> None:
    """Smoothing is the whole point: it must actually remove segments.

    The heart is sampled at 64 points; if simplification were happening in the
    raw coordinate space the tolerance would be ~0 and we would pay one Chime
    message per sample.
    """
    assert len(shape_segments("heart")) < 64


@pytest.mark.parametrize(
    "coarse,fine", [(0.05, 0.02), (0.02, 0.01), (0.01, 0.002)]
)
def test_simplify_is_a_monotonic_cost_dial(coarse: float, fine: float) -> None:
    assert len(shape_segments("heart", simplify=coarse)) <= len(
        shape_segments("heart", simplify=fine)
    )


def test_circle_is_four_segments_and_round() -> None:
    segs = shape_segments("circle", Rect(0.0, 0.0, 1.0, 1.0))
    assert len(segs) == 4
    # Sample the curve and check every point is ~0.5 from the centre.
    for s in segs:
        for i in range(9):
            p = bezier_point(s, i / 8)
            r = math.hypot(p.x - 0.5, p.y - 0.5)
            assert r == pytest.approx(0.5, abs=0.01)


def test_filled_square_costs_more_than_its_outline() -> None:
    outline = shape_segments("square", RECT)
    filled = shape_segments("square", RECT, filled=True, fill_row_height=0.05)
    assert len(filled) > len(outline)


def test_rectangle_is_an_alias_for_square_and_fills() -> None:
    assert shape_segments("rectangle", RECT) == shape_segments("square", RECT)
    assert shape_segments("rect", RECT) == shape_segments("square", RECT)
    filled = shape_segments("rectangle", RECT, filled=True, fill_row_height=0.05)
    assert filled == shape_segments("square", RECT, filled=True, fill_row_height=0.05)


# -- SVG ------------------------------------------------------------------


def test_svg_cubic_is_one_segment() -> None:
    # A single cubic maps 1:1 onto the wire format, so it must not be split.
    assert len(svg_path_segments("M 0 0 C 0.3 1 0.7 1 1 1")) == 1


def test_svg_quadratic_elevates_to_one_cubic() -> None:
    assert len(svg_path_segments("M 0 0 Q 0.5 1 1 0")) == 1


def test_svg_quadratic_elevation_is_exact() -> None:
    """Q -> C elevation must be exact, not an approximation.

    The parser fits its output into the target rect, which applies a uniform
    scale and translation, so this compares shape rather than absolute
    coordinates: the affine map is recovered from the segment's endpoints and
    then every sampled point must agree with the original quadratic.
    """
    seg = svg_path_segments("M 0 0 Q 0.5 1 1 0", Rect(0.0, 0.0, 1.0, 1.0))[0]

    def quad(t: float) -> tuple[float, float]:
        mt = 1 - t
        x = 2 * mt * t * 0.5 + t * t * 1.0
        y = 2 * mt * t * 1.0
        return x, y

    # quad(0) == (0,0) and quad(1) == (1,0); the fit is uniform, so the x
    # endpoints give the scale and both give the offsets.
    scale = seg.d.x - seg.a.x
    assert scale != 0
    ox, oy = seg.a.x, seg.a.y

    for i in range(11):
        t = i / 10
        p = bezier_point(seg, t)
        qx, qy = quad(t)
        assert p.x == pytest.approx(ox + scale * qx, abs=1e-9)
        assert p.y == pytest.approx(oy + scale * qy, abs=1e-9)


def test_svg_relative_and_absolute_agree() -> None:
    a = svg_path_segments("M 0.1 0.1 L 0.5 0.1 L 0.5 0.5")
    b = svg_path_segments("m 0.1 0.1 l 0.4 0 l 0 0.4")
    assert len(a) == len(b)
    for sa, sb in zip(a, b, strict=True):
        assert sa.a.x == pytest.approx(sb.a.x)
        assert sa.d.y == pytest.approx(sb.d.y)


def test_svg_implicit_lineto_after_moveto() -> None:
    assert len(svg_path_segments("M 0 0 0.5 0 1 0")) == 2


def test_svg_close_path_returns_to_start() -> None:
    segs = svg_path_segments(
        "M 0 0 L 1 0 L 1 1 Z", Rect(0.0, 0.0, 1.0, 1.0)
    )
    assert segs[-1].d.x == pytest.approx(segs[0].a.x)
    assert segs[-1].d.y == pytest.approx(segs[0].a.y)


def test_svg_h_and_v() -> None:
    segs = svg_path_segments("M 0 0 H 1 V 1", Rect(0.0, 0.0, 1.0, 1.0))
    assert len(segs) == 2
    assert segs[0].d.y == pytest.approx(segs[0].a.y)  # horizontal
    assert segs[1].d.x == pytest.approx(segs[1].a.x)  # vertical


def test_svg_smooth_cubic_reflects_the_previous_control() -> None:
    segs = svg_path_segments(
        "M 0 0 C 0 1 1 1 1 0 S 2 -1 2 0", Rect(0.0, 0.0, 1.0, 1.0)
    )
    assert len(segs) == 2


def test_svg_arcs_are_rejected_not_silently_dropped() -> None:
    """An arc must raise.

    If the tokenizer dropped A/a instead, its seven numbers would be misread as
    lineto coordinates and produce a silently wrong picture.
    """
    with pytest.raises(ValueError, match="arc"):
        svg_path_segments("M 0 0 A 1 1 0 0 1 1 1")


def test_svg_must_start_with_a_moveto() -> None:
    with pytest.raises(ValueError, match="moveto"):
        svg_path_segments("L 1 1")


def test_svg_odd_coordinate_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="ended mid-command"):
        svg_path_segments("M 0 0 L 1")


def test_svg_empty_is_empty_not_an_error() -> None:
    assert svg_path_segments("") == []


def test_svg_fits_into_the_requested_rect() -> None:
    segs = svg_path_segments("M 0 0 L 10 0 L 10 10 L 0 10 Z", RECT)
    xs = [p.x for p in _all_points(segs)]
    assert min(xs) >= RECT.x0 - EPS
    assert max(xs) <= RECT.x1 + EPS


# -- text -----------------------------------------------------------------


def test_text_in_fits_and_is_non_empty() -> None:
    segs = text_in("HELLO", RECT)
    assert segs
    for p in _all_points(segs):
        assert RECT.x0 - EPS <= p.x <= RECT.x1 + EPS
        assert RECT.y0 - EPS <= p.y <= RECT.y1 + EPS


def test_text_is_uppercased_so_lowercase_input_still_draws() -> None:
    assert len(text_in("hello", RECT)) == len(text_in("HELLO", RECT))


def test_unsupported_characters_render_placeholder_boxes() -> None:
    """Unsupported characters draw a box, they are not silently skipped.

    Worth pinning down: a caller drawing accented text gets a row of boxes and
    pays 4 segments each for them, which is the opposite of free.
    """
    from huddle_drawings.geometry import text_coverage

    assert "É" not in text_coverage()
    segs = text_in("éè", RECT)
    assert len(segs) == 8  # two placeholder boxes, 4 segments each


def test_text_coverage_is_what_the_font_actually_has() -> None:
    from huddle_drawings.geometry import text_coverage

    for ch in "AZ09":
        assert ch in text_coverage()
        assert text_in(ch, RECT)


def test_longer_text_costs_more() -> None:
    assert len(text_in("AA", RECT)) > len(text_in("A", RECT))


def _curve_extent(segs):
    """Where the drawn curves actually reach, not their control points."""
    def at(s, t):
        u = 1 - t
        w = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
        return (sum(k * q.x for k, q in zip(w, s.points)),
                sum(k * q.y for k, q in zip(w, s.points)))

    pts = [at(s, t / 20) for s in segs for t in range(21)]
    return (min(x for x, _ in pts), min(y for _, y in pts),
            max(x for x, _ in pts), max(y for _, y in pts))


@pytest.mark.parametrize(
    "target",
    [Rect(0.0, 0.0, 1.0, 1.0), Rect(0.1, 0.6, 0.9, 0.9), Rect(0.2, 0.1, 0.4, 0.8)],
)
def test_filled_rectangle_covers_exactly_its_rect(target: Rect) -> None:
    # A hairpin's controls overshoot the edge so its tip lands on it; fitting by
    # control points used to shrink a full-screen fill to the middle 3/4.
    x0, y0, x1, y1 = _curve_extent(shape_segments("rectangle", target, filled=True))
    assert x0 == pytest.approx(target.x0, abs=1e-3)
    assert x1 == pytest.approx(target.x1, abs=1e-3)
    assert y0 == pytest.approx(target.y0, abs=1e-3)
    assert y1 == pytest.approx(target.y1, abs=0.01)  # last band may stop a pen short


def test_rectangle_outline_is_not_squared_off() -> None:
    target = Rect(0.1, 0.6, 0.9, 0.9)
    assert _curve_extent(shape_segments("rectangle", target)) == pytest.approx(
        (0.1, 0.6, 0.9, 0.9)
    )


def test_square_stays_square_and_centred() -> None:
    x0, y0, x1, y1 = _curve_extent(shape_segments("square", Rect(0.0, 0.0, 1.0, 0.5)))
    assert (x0, y0, x1, y1) == pytest.approx((0.25, 0.0, 0.75, 0.5))
