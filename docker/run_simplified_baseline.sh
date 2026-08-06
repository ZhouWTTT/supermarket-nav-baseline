#!/usr/bin/env bash
set -eo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
cd "$repo_root"

# Choose run mode:
#   nav     — navigation-only task: shelf D → shelf B → delivery (default)
#   grasp   — fixed baseline: vision → pick → retreat (requires kele.pt)
MODE="${BASELINE_MODE:-nav}"

if [ "$MODE" = "nav" ]; then
    echo "=== Running navigation demo (no perception/grasping) ==="
    python3 examples/supermarket_sorting/supermarket_navigation_demo.py
else
    echo "=== Running fixed grasp baseline ==="
    python3 examples/supermarket_sorting/perception/kele_detect.py \
      --backend "${SUPERMARKET_DETECTOR_BACKEND:-yolo}" \
      --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
    detector_pid=$!

    cleanup() {
      kill "$detector_pid" 2>/dev/null || true
      wait "$detector_pid" 2>/dev/null || true
    }
    trap cleanup EXIT

    python3 examples/supermarket_sorting/supermarket_sorting_client_simplified.py
fi
