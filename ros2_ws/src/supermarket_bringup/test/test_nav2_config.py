from pathlib import Path
import re


CONFIG = Path(__file__).parents[1] / "config" / "nav2_candidate.yaml"


def _section(name: str, next_name: str) -> str:
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(
        rf"^    {name}:\n(?P<body>.*?)(?=^    {next_name}:)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_humble_approach_polygon_cannot_use_zero_point_threshold():
    """Nav2 Humble treats 0 >= max_points as an immediate collision."""

    section = _section("FootprintApproach", "observation_sources")
    match = re.search(r"^      max_points: (\d+)$", section, re.MULTILINE)
    assert match is not None
    assert int(match.group(1)) > 0


def test_collision_monitor_is_the_only_final_velocity_output():
    text = CONFIG.read_text(encoding="utf-8")
    section = text.split("collision_monitor:\n", 1)[1]
    assert "cmd_vel_in_topic: /motion/smoothed_cmd_vel" in section
    assert "cmd_vel_out_topic: /cmd_vel" in section


def test_candidate_pick_contract_defaults_middle_tissue_grasp_off():
    executive = (
        CONFIG.parents[2]
        / "supermarket_mission"
        / "supermarket_mission"
        / "mission_executive.py"
    ).read_text(encoding="utf-8")
    assert 'declare_parameter("enable_zhijin_middle_column", False)' in executive
