"""Tests for the pure geometry layer: curves, simplification, hairpins, shapes."""

from __future__ import annotations

import math

import pytest

from huddle_drawings import geometry as g

CURVE_POINTS = [
    g.Point(0.10, 0.10),
    g.Point(0.25, 0.40),
    g.Point(0.50, 0.15),
    g.Point(0.70, 0.60),
    g.Point(0.90, 0.30),
]


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def point_to_segment_distance(p: g.Point, a: g.Point, b: g.Point) -> float:
    """Distance from ``p`` to the finite segment ``a``-``b``."""
    dx, dy = b.x - a.x, b.y - a.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return p.distance_to(a)
    t = max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length_sq))
    return p.distance_to(g.Point(a.x + t * dx, a.y + t * dy))


def distance_to_polyline(p: g.Point, polyline: list[g.Point]) -> float:
    return min(
        point_to_segment_distance(p, a, b)
        for a, b in zip(polyline, polyline[1:], strict=False)
    )


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------


def test_point_arithmetic_and_unpacking() -> None:
    p = g.Point(0.2, 0.4)
    x, y = p
    assert (x, y) == (0.2, 0.4)
    assert p + g.Point(0.1, 0.1) == g.Point(0.30000000000000004, 0.5)
    assert (p - p) == g.Point(0.0, 0.0)
    assert (p * 2).x == 0.4
    assert (2 * p).y == 0.8


def test_point_of_accepts_tuples_and_protobuf() -> None:
    from huddle_drawings.proto import peer_message_pb2 as pb

    assert g.Point.of((0.1, 0.2)) == g.Point(0.1, 0.2)
    assert g.Point.of([0.3, 0.4]) == g.Point(0.3, 0.4)
    assert g.Point.of(g.Point(0.5, 0.6)) == g.Point(0.5, 0.6)
    proto = pb.PeerMessage.Point(x=0.7, y=0.8)
    assert g.Point.of(proto) == g.Point(0.7, 0.8)


def test_clamp01_and_clamped() -> None:
    assert g.clamp01(-5.0) == 0.0
    assert g.clamp01(1.5) == 1.0
    assert g.clamp01(0.5) == 0.5
    assert g.Point(-5, 2).clamped() == g.Point(0.0, 1.0)
    seg = g.Segment.line(g.Point(-1, -1), g.Point(2, 2)).clamped()
    assert all(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in seg.points)


def test_degenerate_segment_is_the_first_point_of_a_stroke() -> None:
    seg = g.Segment.degenerate((0.3, 0.7))
    assert seg.a == seg.b == seg.c == seg.d == g.Point(0.3, 0.7)
    assert seg.is_degenerate


def test_straight_segment_is_actually_straight() -> None:
    seg = g.Segment.line((0.0, 0.0), (1.0, 0.5))
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = g.bezier_point(seg, t)
        assert close(p.y, p.x * 0.5)
    assert not seg.is_degenerate


def test_segment_reversed_traces_the_same_curve() -> None:
    seg = g.Segment(g.Point(0, 0), g.Point(0.2, 0.9), g.Point(0.8, 0.9), g.Point(1, 0))
    rev = seg.reversed()
    for t in (0.0, 0.3, 0.5, 1.0):
        forward = g.bezier_point(seg, t)
        backward = g.bezier_point(rev, 1 - t)
        assert close(forward.x, backward.x)
        assert close(forward.y, backward.y)


def test_proto_round_trip() -> None:
    seg = g.Segment(
        g.Point(0.1, 0.2), g.Point(0.3, 0.4), g.Point(0.5, 0.6), g.Point(0.7, 0.8)
    )
    proto = g.to_proto_segment(seg)
    assert (proto.a.x, proto.a.y) == (0.1, 0.2)
    assert (proto.d.x, proto.d.y) == (0.7, 0.8)
    assert g.from_proto_segment(proto) == seg
    assert len(g.to_proto_segments([seg, seg])) == 2


def test_pen_size_normalized() -> None:
    w, h = g.pen_size_normalized(1920, 1080)
    assert close(w, 7 / 1920)
    assert close(h, 7 / 1080)
    with pytest.raises(ValueError):
        g.pen_size_normalized(0, 100)


# --------------------------------------------------------------------------
# Catmull-Rom
# --------------------------------------------------------------------------


def test_catmull_rom_passes_through_every_input_point() -> None:
    segments = g.catmull_rom_to_beziers(CURVE_POINTS)
    assert len(segments) == len(CURVE_POINTS) - 1
    for i, seg in enumerate(segments):
        start = g.bezier_point(seg, 0.0)
        end = g.bezier_point(seg, 1.0)
        assert close(start.x, CURVE_POINTS[i].x) and close(start.y, CURVE_POINTS[i].y)
        assert close(end.x, CURVE_POINTS[i + 1].x) and close(
            end.y, CURVE_POINTS[i + 1].y
        )
        # The anchors are the input points verbatim.
        assert seg.a == CURVE_POINTS[i]
        assert seg.d == CURVE_POINTS[i + 1]


def test_catmull_rom_is_continuous_between_segments() -> None:
    segments = g.catmull_rom_to_beziers(CURVE_POINTS)
    for first, second in zip(segments, segments[1:], strict=False):
        assert first.d == second.a


def test_catmull_rom_closed_adds_a_closing_segment() -> None:
    square = [g.Point(0.2, 0.2), g.Point(0.8, 0.2), g.Point(0.8, 0.8), g.Point(0.2, 0.8)]
    closed = g.catmull_rom_to_beziers(square, closed=True)
    assert len(closed) == len(square)
    assert closed[-1].d == square[0]


def test_catmull_rom_curves_away_from_the_straight_line() -> None:
    """A curve buys more picture than a straight dash -- it must actually bend."""
    segments = g.catmull_rom_to_beziers(CURVE_POINTS)
    middle = segments[1]
    chord = g.Segment.line(middle.a, middle.d)
    bend = max(
        g.bezier_point(middle, t).distance_to(g.bezier_point(chord, t))
        for t in (0.25, 0.5, 0.75)
    )
    assert bend > 0.01


def test_catmull_rom_degenerate_inputs() -> None:
    assert g.catmull_rom_to_beziers([]) == []
    assert g.catmull_rom_to_beziers([g.Point(0.5, 0.5)]) == []
    two = g.catmull_rom_to_beziers([g.Point(0.1, 0.1), g.Point(0.2, 0.2)])
    assert len(two) == 1
    # Repeated points collapse rather than producing NaNs.
    repeated = g.catmull_rom_to_beziers([g.Point(0.1, 0.1)] * 5)
    assert repeated == []


def test_zero_tension_gives_straight_lines() -> None:
    segments = g.catmull_rom_to_beziers(CURVE_POINTS, tension=0.0)
    for seg in segments:
        # Controls collapse onto the anchors, so the curve traces the chord.
        assert seg.b == seg.a
        assert seg.c == seg.d
        for t in (0.25, 0.5, 0.75):
            p = g.bezier_point(seg, t)
            assert point_to_segment_distance(p, seg.a, seg.d) < 1e-12


# --------------------------------------------------------------------------
# RDP
# --------------------------------------------------------------------------


def test_rdp_reduces_a_nearly_straight_polyline_to_its_endpoints() -> None:
    points = [g.Point(i / 100, 0.5 + 0.0005 * math.sin(i)) for i in range(101)]
    simplified = g.rdp(points, 0.01)
    assert len(simplified) == 2
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]


def test_rdp_actually_reduces_point_count_but_keeps_the_shape() -> None:
    # A zig-zag sampled far too finely, as a traced contour would be.
    points: list[g.Point] = []
    for i in range(200):
        t = i / 199
        points.append(g.Point(t, 0.5 + 0.2 * math.sin(t * 6 * math.pi)))
    simplified = g.rdp(points, 0.01)
    assert 5 < len(simplified) < len(points) / 4
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]
    # RDP's guarantee: every dropped point is within epsilon of the polyline
    # that was kept, so the shape survives even though the point count does not.
    for p in points:
        assert distance_to_polyline(p, simplified) <= 0.01 + 1e-9


def test_rdp_keeps_sharp_corners() -> None:
    points = [g.Point(0, 0), g.Point(0.5, 0), g.Point(0.5, 0.5), g.Point(1, 0.5)]
    assert g.rdp(points, 0.001) == points


def test_rdp_short_inputs_and_validation() -> None:
    assert g.rdp([], 0.1) == []
    assert len(g.rdp([g.Point(0, 0), g.Point(1, 1)], 0.1)) == 2
    with pytest.raises(ValueError):
        g.rdp(CURVE_POINTS, -1.0)


def test_smooth_path_is_simplify_then_fit() -> None:
    points = [g.Point(i / 200, 0.5 + 0.25 * math.sin(i / 20)) for i in range(201)]
    segments = g.smooth_path(points, epsilon=0.01)
    assert 0 < len(segments) < 20  # ~200 straight edges would be ~200 messages
    assert segments[0].a == points[0]
    assert segments[-1].d == points[-1]


# --------------------------------------------------------------------------
# hairpins
# --------------------------------------------------------------------------


def test_hairpin_matches_the_control_points_the_doc_describes() -> None:
    """``a=(x0,y), b=(x1',y), c=(x1',y+1), d=(x0,y+1)`` -- out and back."""
    x0, x1, y, dy = 0.2, 0.8, 0.3, 0.01
    seg = g.hairpin_segment(x0, x1, y, dy)
    tip = x0 + (x1 - x0) * g.HAIRPIN_OVERSHOOT

    assert seg.a == g.Point(x0, y)                 # starts at the row's left end
    assert close(seg.b.x, tip) and close(seg.b.y, y)       # out along row y
    assert close(seg.c.x, tip) and close(seg.c.y, y + dy)  # back along row y+1
    assert seg.b.x == seg.c.x                      # both controls at x1'
    assert close(seg.d.x, x0) and close(seg.d.y, y + dy)   # ends under the start


def test_hairpin_tip_reaches_the_requested_far_edge() -> None:
    seg = g.hairpin_segment(0.1, 0.9, 0.2, 0.01)
    apex = g.bezier_point(seg, 0.5)
    assert close(apex.x, 0.9)
    assert close(apex.y, 0.205)  # halfway between the two rows
    # And it never goes past it.
    assert max(g.bezier_point(seg, i / 100).x for i in range(101)) <= 0.9 + 1e-12


def test_hairpin_fills_a_band_two_rows_tall() -> None:
    dy = 0.01
    seg = g.hairpin_segment(0.0, 1.0, 0.5, dy)
    ys = [g.bezier_point(seg, i / 200).y for i in range(201)]
    assert close(min(ys), 0.5)
    assert close(max(ys), 0.5 + dy)
    # Monotonic descent from row y to row y+dy: no wasted vertical travel.
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:], strict=False))


def test_hairpin_without_overshoot_puts_controls_exactly_on_x1() -> None:
    seg = g.hairpin_segment(0.0, 1.0, 0.0, 0.01, overshoot=1.0)
    assert close(seg.b.x, 1.0)
    assert close(g.bezier_point(seg, 0.5).x, 0.75)  # the short tip


def test_hairpin_fill_halves_the_rows_needed() -> None:
    rect = g.Rect(0.1, 0.1, 0.9, 0.3)
    row = 0.01
    segments = g.hairpin_fill(rect, row)
    plain_hatching_rows = rect.height / row
    # One message per two pen rows, per the doc.
    assert len(segments) == pytest.approx(plain_hatching_rows / 2, abs=1)
    for seg in segments:
        for p in seg.points:
            assert rect.y0 - 1e-9 <= p.y <= rect.y1 + 1e-9
    # Bands step by two row heights.
    assert close(segments[1].a.y - segments[0].a.y, 2 * row)


def test_hairpin_fill_serpentine_alternates_direction() -> None:
    rect = g.Rect(0.0, 0.0, 1.0, 0.1)
    flipped = g.hairpin_fill(rect, 0.01, serpentine=True)
    assert flipped[0].a.x == 0.0
    assert flipped[1].a.x == 1.0
    straight = g.hairpin_fill(rect, 0.01, serpentine=False)
    assert all(seg.a.x == 0.0 for seg in straight)


def test_hairpin_fill_validates_row_height() -> None:
    with pytest.raises(ValueError):
        g.hairpin_fill(g.unit_square(), 0.0)


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def inside_unit_square(segments) -> bool:
    return all(
        0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for s in segments for p in s.points
    )


def test_line_is_one_message() -> None:
    segments = g.line_segments((0.1, 0.1), (0.9, 0.9))
    assert len(segments) == 1
    assert inside_unit_square(segments)


def test_polyline_is_one_segment_per_edge() -> None:
    pts = [g.Point(0.1, 0.1), g.Point(0.5, 0.2), g.Point(0.9, 0.8)]
    assert len(g.polyline_segments(pts)) == 2
    assert len(g.polyline_segments(pts, closed=True)) == 3
    assert g.polyline_segments([g.Point(0.5, 0.5)]) == []


def test_rect_outline_and_fill_stay_in_bounds() -> None:
    rect = g.Rect(0.2, 0.2, 0.8, 0.6)
    outline = g.rect_segments(rect)
    assert len(outline) == 4
    assert inside_unit_square(outline)
    corners = {(p.x, p.y) for s in outline for p in (s.a, s.d)}
    assert corners == {(0.2, 0.2), (0.8, 0.2), (0.8, 0.6), (0.2, 0.6)}

    filled = g.rect_segments(rect, fill_row_height=0.02)
    assert len(filled) > 1
    # The hairpin control points overshoot sideways by design; the visible
    # curve stays inside.
    assert all(rect.y0 <= p.y <= rect.y1 for s in filled for p in s.points)


def test_circle_and_ellipse() -> None:
    segments = g.circle_segments((0.5, 0.5), 0.25)
    assert len(segments) == 4
    assert inside_unit_square(segments)
    # Sampled points sit on the circle (kappa approximation, ~0.03% error).
    for seg in segments:
        for i in range(11):
            p = g.bezier_point(seg, i / 10)
            assert abs(math.hypot(p.x - 0.5, p.y - 0.5) - 0.25) < 0.001
    assert segments[0].a == segments[-1].d  # closed

    quarter = g.ellipse_segments((0.5, 0.5), 0.3, 0.1, quarters=1)
    assert len(quarter) == 1
    with pytest.raises(ValueError):
        g.ellipse_segments((0.5, 0.5), 0.1, 0.1, quarters=5)


def test_text_segments_stay_inside_their_box() -> None:
    segments = g.text_segments("HELLO 42", (0.1, 0.1), height=0.1)
    assert segments
    assert inside_unit_square(segments)
    xs = [p.x for s in segments for p in s.points]
    ys = [p.y for s in segments for p in s.points]
    assert min(xs) >= 0.1 - 1e-9
    assert min(ys) >= 0.1 - 1e-9
    assert max(ys) <= 0.1 + 0.1 + 1e-9  # one line tall


def test_text_covers_the_documented_characters() -> None:
    coverage = g.text_coverage()
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-_:!?/'+=()":
        assert char in coverage
    # Lowercase is upper-cased, not dropped.
    assert g.text_segments("abc", height=0.1) == g.text_segments("ABC", height=0.1)
    # A space draws nothing but still advances.
    assert g.text_segments(" ", height=0.1) == []
    # An unknown character is drawn as a visible box, not silently skipped.
    assert len(g.text_segments("§", height=0.1)) == 4


def test_text_newline_starts_a_new_line() -> None:
    one = g.text_segments("A", (0.0, 0.0), height=0.1)
    two = g.text_segments("A\nA", (0.0, 0.0), height=0.1)
    assert len(two) == 2 * len(one)
    assert max(p.y for s in two for p in s.points) > 0.1


def test_text_curved_mode_uses_fewer_segments_for_round_glyphs() -> None:
    straight = g.text_segments("O", height=0.2)
    curved = g.text_segments("O", height=0.2, curved=True)
    assert len(curved) <= len(straight)


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_fit_points_into_a_corner() -> None:
    corner = g.Rect(0.75, 0.75, 1.0, 1.0)
    points = [g.Point(0, 0), g.Point(10, 10), g.Point(5, 2)]
    fitted = g.fit_points_into_rect(points, corner)
    assert all(corner.contains(p) for p in fitted)
    assert fitted[0] == g.Point(0.75, 0.75)
    assert fitted[1] == g.Point(1.0, 1.0)


def test_fit_preserves_aspect_and_centres() -> None:
    target = g.Rect(0.0, 0.0, 1.0, 0.5)
    wide = [g.Point(0, 0), g.Point(2, 1)]
    fitted = g.fit_points_into_rect(wide, target)
    assert close(fitted[0].x, 0.0)
    assert close(fitted[1].x, 1.0)
    # Centred vertically in the slack.
    assert close(fitted[0].y, 0.0) and close(fitted[1].y, 0.5)

    stretched = g.fit_points_into_rect(wide, target, preserve_aspect=False)
    assert close(stretched[1].y, 0.5)


def test_fit_segments_keeps_curves_inside_the_rect() -> None:
    circle = g.circle_segments((0.5, 0.5), 0.5)
    corner = g.Rect(0.6, 0.05, 0.95, 0.4)
    fitted = g.fit_segments_into_rect(circle, corner)
    assert len(fitted) == len(circle)
    # A Bézier stays within the convex hull of its control points, so bounding
    # the control points bounds the drawing.
    assert all(corner.contains(p) for s in fitted for p in s.points)


def test_fit_handles_degenerate_input() -> None:
    assert g.fit_points_into_rect([], g.unit_square()) == []
    assert g.fit_segments_into_rect([], g.unit_square()) == []
    vertical = g.fit_points_into_rect(
        [g.Point(0.5, 0.0), g.Point(0.5, 1.0)], g.Rect(0.0, 0.0, 0.5, 0.5)
    )
    assert all(0.0 <= p.y <= 0.5 for p in vertical)


def test_rect_helpers() -> None:
    rect = g.Rect(0.8, 0.9, 0.2, 0.1).normalized()
    assert (rect.x0, rect.y0, rect.x1, rect.y1) == (0.2, 0.1, 0.8, 0.9)
    assert close(rect.width, 0.6) and close(rect.height, 0.8)
    assert rect.center == g.Point(0.5, 0.5)
    assert rect.contains((0.5, 0.5))
    assert not rect.contains((0.0, 0.0))
    assert g.Rect.bounding(CURVE_POINTS).x0 == 0.10
    assert g.unit_square() == g.Rect(0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        g.Rect.bounding([])


def test_geometry_does_not_import_protobuf_at_module_level() -> None:
    """The doc's drawing math should stand alone from the wire format."""
    import ast
    import pathlib

    source = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:  # top level only; lazy imports live inside functions
        if isinstance(node, ast.ImportFrom):
            assert "proto" not in (node.module or "")
        elif isinstance(node, ast.Import):
            assert all("proto" not in alias.name for alias in node.names)
