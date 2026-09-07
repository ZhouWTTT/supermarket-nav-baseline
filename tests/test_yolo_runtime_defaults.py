"""Keep every supported launcher on the formal YOLO runtime defaults."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _argument_default(path: str, option: str):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for call in (
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"):
        if not call.args:
            continue
        first = call.args[0]
        if not isinstance(first, ast.Constant) or first.value != option:
            continue
        default = next(
            keyword.value for keyword in call.keywords
            if keyword.arg == "default")
        return ast.literal_eval(default)
    raise AssertionError(f"missing argument {option} in {path}")


def test_formal_runner_defaults_to_cuda_point9_and_8hz():
    path = "examples/supermarket_sorting/competition_runner.py"
    assert _argument_default(path, "--device") == "cuda"
    assert _argument_default(path, "--confidence") == 0.90
    assert _argument_default(path, "--inference-hz") == 8.0


def test_persistent_perception_has_the_same_defaults():
    path = "examples/supermarket_sorting/persistent_perception.py"
    assert _argument_default(path, "--device") == "cuda"
    assert _argument_default(path, "--confidence") == 0.90
    assert _argument_default(path, "--max-inference-hz") == 8.0


def test_integrated_worker_has_the_same_defaults():
    path = "examples/supermarket_sorting/integrated_nav_pick_place.py"
    assert _argument_default(path, "--device") == "cuda"
    assert _argument_default(path, "--confidence") == 0.90
    assert _argument_default(path, "--max-inference-hz") == 8.0


def test_baseline_script_explicitly_forwards_the_defaults():
    source = (ROOT / "scripts/run_baseline.sh").read_text(encoding="utf-8")
    assert '${SUPERMARKET_DEVICE:-cuda}' in source
    assert '${SUPERMARKET_YOLO_CONFIDENCE:-0.90}' in source
    assert '${SUPERMARKET_INFERENCE_HZ:-8}' in source
