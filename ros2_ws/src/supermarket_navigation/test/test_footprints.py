import pytest

from supermarket_navigation.footprint_manager import PROFILES, PROFILE_DOCUMENT


def bounds(profile):
    xs = [point[0] for point in profile.points]
    ys = [point[1] for point in profile.points]
    return min(xs), max(xs), min(ys), max(ys)


def test_no_profile_shrinks_below_physical_chassis_seed():
    compact = bounds(PROFILES["COMPACT_TRANSIT"])
    for profile in PROFILES.values():
        current = bounds(profile)
        assert current[0] <= compact[0]
        assert current[1] >= compact[1]
        assert current[2] <= compact[2]
        assert current[3] >= compact[3]


def test_extended_profile_is_larger_than_transit_profile():
    compact = bounds(PROFILES["COMPACT_TRANSIT"])
    extended = bounds(PROFILES["MANIPULATION_EXTENDED"])
    assert extended[0] < compact[0]
    assert extended[1] > compact[1]
    assert extended[2] < compact[2]
    assert extended[3] > compact[3]


def test_compact_profile_contains_generated_collision_geometry_plus_margin():
    raw_xmin, raw_xmax, raw_ymin, raw_ymax = PROFILE_DOCUMENT[
        "raw_chassis_bounds"
    ]
    margin = PROFILE_DOCUMENT["margin_m"]
    compact = bounds(PROFILES["COMPACT_TRANSIT"])
    assert compact == pytest.approx(
        (
            raw_xmin - margin,
            raw_xmax + margin,
            raw_ymin - margin,
            raw_ymax + margin,
        ),
        abs=1.0e-4,
    )
