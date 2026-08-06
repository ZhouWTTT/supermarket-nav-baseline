#!/usr/bin/env python3
"""
Pluggable 2-D detector backends for the kele perception pipeline.

Each backend exposes a single method::

    detect(rgb, depth, K, T_cam_world=None) -> list[dict]

where every dict has::

    class  : str   – always 'kele' for all backends here
    x, y   : int   – pixel coords of the bbox centre
    w, h   : int   – bbox size in pixels
    conf   : float – confidence in [0, 1]

The downstream node does pixel→camera-frame deprojection and the
camera→world transform.  Backends do NOT need to know the world frame.

Exception: GtProjectionBackend (for coord-bridge validation) also
accepts T_cam_world to project GT world positions to pixels; it sets
an extra 'gt_world_pos' key in each result so the node can log the
round-trip error.

Usage (in the node):
    from perception.backends import GtProjectionBackend, BlobBackend, YoloBackend
    detector = BlobBackend()                       # black-background scene
    detector = GtProjectionBackend(layout_path)    # coord-bridge validation
    detector = YoloBackend(ckpt_path)              # final trained model
"""

import os
import json
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_CLASS = 'kele'


def _safe_depth_m(depth_img: np.ndarray, cx: int, cy: int, r: int = 4) -> float:
    """Median depth (metres) of a square patch centred on (cx,cy), ignoring zeros."""
    h, w = depth_img.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = depth_img[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) * 1e-3 if len(valid) > 0 else 0.0


# ---------------------------------------------------------------------------
# GtProjectionBackend
# ---------------------------------------------------------------------------
class GtProjectionBackend:
    """Validate the coordinate bridge.

    Projects each kele slot's GT world position to pixels via T_cam_world,
    then the node's pixel→world path should reconstruct the same point.
    Attach 'gt_world_pos' to each detection dict for logging.

    This backend requires T_cam_world to be passed in (not None).
    """

    def __init__(self, layout_path: str):
        with open(layout_path, 'r') as f:
            layout = json.load(f)
        # keep only kele entries
        self.kele_slots = [s for s in layout if s.get('object_kind') == 'kele']

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        """Project GT kele positions to pixels; return in-frame detections."""
        if T_cam_world is None:
            return []  # can't project without the transform

        H, W = rgb.shape[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        T_world_cam = np.linalg.inv(T_cam_world)  # world → camera

        detections = []
        for slot in self.kele_slots:
            pw = np.array(slot['world_position'] + [1.0], dtype=float)
            pc = T_world_cam @ pw   # camera frame
            if pc[2] <= 0.05:       # behind or too close
                continue
            u = int(fx * pc[0] / pc[2] + cx)
            v = int(fy * pc[1] / pc[2] + cy)
            if not (0 <= u < W and 0 <= v < H):
                continue            # out of frame
            # fake bbox (10 × 10 px around projection)
            bbox_px = 20
            detections.append({
                'class': _CLASS,
                'x': u,
                'y': v,
                'w': bbox_px,
                'h': bbox_px,
                'conf': 1.0,
                'gt_world_pos': np.array(slot['world_position']),
                'body': slot['body'],
            })
        return detections


# ---------------------------------------------------------------------------
# BlobBackend
# ---------------------------------------------------------------------------
class BlobBackend:
    """Find non-black object blobs in the GS-rendered image.

    With a black background, every Gaussian-splatted object renders as a
    non-black pixel cluster.  Simple thresholding + connected components
    gives reliable bboxes with no model weights needed.

    Parameters
    ----------
    min_area   : minimum blob area in pixels (filters noise)
    max_area   : maximum blob area (filters full-frame glare)
    value_thr  : HSV value threshold (pixels brighter than this are "object")
    depth_min  : ignore detections with depth < this (metres; filters close noise)
    depth_max  : ignore detections with depth > this (metres; shelf is ~0.5 m away)
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 80_000,
        value_thr: int = 15,
        depth_min: float = 0.15,
        depth_max: float = 2.50,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.value_thr = value_thr
        self.depth_min = depth_min
        self.depth_max = depth_max

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        # threshold: keep pixels with V > value_thr in HSV
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 2] > self.value_thr).astype(np.uint8) * 255

        # morphological cleanup: close small holes, open stray pixels
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detections = []
        for i in range(1, n_labels):  # skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            cx_px = int(centroids[i, 0])
            cy_px = int(centroids[i, 1])
            w_px  = int(stats[i, cv2.CC_STAT_WIDTH])
            h_px  = int(stats[i, cv2.CC_STAT_HEIGHT])
            # depth gate: only accept blobs at reasonable shelf distance
            d = _safe_depth_m(depth, cx_px, cy_px)
            if d < self.depth_min or d > self.depth_max:
                continue
            detections.append({
                'class': _CLASS,
                'x': cx_px,
                'y': cy_px,
                'w': w_px,
                'h': h_px,
                'conf': 0.80,   # nominal confidence for blob detections
            })
        return detections


# ---------------------------------------------------------------------------
# YoloBackend  (drop-in final backend once kele.pt is trained)
# ---------------------------------------------------------------------------
class YoloBackend:
    """YOLOv8 detector (ultralytics).  Mirrors the reference yolo_detect.py
    but stripped to single-class 'kele'.

    If the checkpoint file does not exist yet the backend logs a warning and
    returns an empty list (graceful degradation; swap to BlobBackend instead).
    """

    CLASS_NAMES = ['kele']

    def __init__(self, ckpt_path: str, conf_thresh: float = 0.65, device: str = "auto"):
        self.conf_thresh = conf_thresh
        self.model = None
        self.device = device
        if not os.path.isfile(ckpt_path):
            print(f"[YoloBackend] checkpoint not found: {ckpt_path} — returning empty detections")
            return
        try:
            import torch
            from ultralytics import YOLO

            device = self._select_device(torch, device)
            # Patch torch.load for pre-2.6 checkpoints
            _orig = torch.load
            def _compat(*a, **kw):
                kw.setdefault('weights_only', False)
                return _orig(*a, **kw)
            torch.load = _compat
            try:
                self.model = YOLO(ckpt_path).to(device)
                self.model.model.eval()
            finally:
                torch.load = _orig
            print(f"[YoloBackend] loaded {ckpt_path} on {device}")
        except Exception as e:
            print(f"[YoloBackend] failed to load model: {e}")

    @staticmethod
    def _select_device(torch, requested: str):
        requested = requested.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("YOLO device must be auto, cpu, or cuda")
        if requested == "cpu":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            print("[YoloBackend] CUDA unavailable; using CPU")
            return torch.device("cpu")

        major, minor = torch.cuda.get_device_capability(0)
        supported = []
        for arch in torch.cuda.get_arch_list():
            if arch.startswith("sm_"):
                supported.append(int(arch[3:]))
        capability = major * 10 + minor
        compatible = any(arch // 10 == major and arch % 10 <= minor for arch in supported)
        if compatible:
            return torch.device("cuda:0")

        print(
            f"[YoloBackend] GPU sm_{capability} is not supported by this PyTorch build "
            f"({torch.cuda.get_arch_list()}); using CPU"
        )
        return torch.device("cpu")

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cam_world: np.ndarray | None = None,
    ) -> list[dict]:
        if self.model is None:
            return []
        results = self.model(rgb, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.conf_thresh:
                continue
            cls_id = int(box.cls.item())
            if cls_id >= len(self.CLASS_NAMES):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            detections.append({
                'class': self.CLASS_NAMES[cls_id],
                'x': (x0 + x1) // 2,
                'y': (y0 + y1) // 2,
                'w': x1 - x0,
                'h': y1 - y0,
                'conf': conf,
            })
        return detections
