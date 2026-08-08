#!/usr/bin/env python3
"""Fine-tune YOLOv8s on the generated kele dataset (single class).

Runs INSIDE the Docker container (needs ultralytics + a GPU).  Expects the
dataset produced by gen_dataset.py at perception/dataset/.  After training it
copies the best weights to perception/checkpoints/kele.pt and prints the
validation mAP so the caller can sanity-check the run.

Run (inside container):
    cd examples/supermarket_sorting
    python3 perception/train_yolo.py --epochs 100
"""
import os
import argparse
import shutil
from pathlib import Path

PERCEPTION_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PERCEPTION_DIR / "dataset" / "data.yaml"
CKPT_DIR = PERCEPTION_DIR / "checkpoints"
FINAL_CKPT = CKPT_DIR / "kele.pt"


def main():
    ap = argparse.ArgumentParser(description="fine-tune YOLOv8s for kele")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="pretrained weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", default=-1,
                    help="batch size (-1 = ultralytics auto)")
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop patience (epochs)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", default="false",
                    choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
                    help="enable AMP training checks; false avoids downloading yolov8n.pt")
    ap.add_argument("--project", default=str(PERCEPTION_DIR / "runs"))
    ap.add_argument("--name", default="kele_yolov8s")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    # ultralytics 8.0.196 + torch>=2.6 needs weights_only=False for old ckpts
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat

    try:
        batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch
        model = YOLO(args.weights)
        amp = str(args.amp).lower() in {"true", "1", "yes", "on"}
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            patience=args.patience,
            device=args.device,
            amp=amp,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
        metrics = model.val()
    finally:
        torch.load = _orig

    # locate best.pt from the run
    best = Path(args.project) / args.name / "weights" / "best.pt"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        shutil.copy2(best, FINAL_CKPT)
        print(f"[train] copied {best} -> {FINAL_CKPT}")
    else:
        print(f"[train] WARNING: best.pt not found at {best}")

    try:
        print(f"[train] val mAP50-95 = {metrics.box.map:.4f}")
        print(f"[train] val mAP50    = {metrics.box.map50:.4f}")
    except Exception as e:
        print(f"[train] could not read metrics: {e}")


if __name__ == "__main__":
    main()
