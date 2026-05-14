"""
Thin ONNX-only inference wrapper that replaces ultralytics.YOLO for runtime.
Requires only onnxruntime + numpy — no torch, no ultralytics.

Supports:
  - YOLOv8 detection models  (output shape: B × (4+num_cls) × anchors)
  - YOLOv8 pose models       (output shape: B × (4+1+17×3) × anchors)

Usage is intentionally compatible with the ultralytics YOLO call convention so
existing code needs only to change the import, not the inference calls.
"""

import cv2
import numpy as np
import onnxruntime as ort

# COCO 80-class names in index order
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

NUM_POSE_KP = 17  # COCO keypoints


# ---------------------------------------------------------------------------
# Small data-holder objects that mimic the ultralytics result API
# ---------------------------------------------------------------------------

class _TensorLike:
    """Wraps a numpy array so that .cpu().numpy() works unchanged."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)


class _Keypoints:
    def __init__(self, xy: np.ndarray):
        # xy shape: (N_persons, 17, 2)
        self.xy = _TensorLike(xy)


class _Box:
    def __init__(self, xyxy: np.ndarray, cls_id: int, conf: float):
        self.xyxy = [xyxy]           # xyxy[0] → np.ndarray([x1,y1,x2,y2])
        self.cls = [float(cls_id)]   # cls[0]  → class id
        self.conf = [float(conf)]


class _Boxes:
    def __init__(self, box_list: list):
        self._boxes = box_list

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class _Result:
    def __init__(self, boxes: _Boxes = None, keypoints: _Keypoints = None):
        self.boxes = boxes if boxes is not None else _Boxes([])
        self.keypoints = keypoints  # None for detection models


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, target_hw: tuple):
    """Resize + pad to target (H, W) while keeping aspect ratio."""
    h, w = img.shape[:2]
    th, tw = target_hw
    ratio = min(th / h, tw / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    img = cv2.copyMakeBorder(
        img,
        pad_top, th - new_h - pad_top,
        pad_left, tw - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img, ratio, pad_left, pad_top


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class YOLOOnnx:
    """
    Drop-in ONNX replacement for ultralytics.YOLO when the model is already
    exported to .onnx.  Supports YOLOv8 detection and pose models.

    Example
    -------
    model = YOLOOnnx("yolov8n.onnx")
    results = model(frame, conf=0.5)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
    """

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self._input_name = inp.name

        # Read the fixed input spatial size from the model itself
        shape = inp.shape  # e.g. [1, 3, 640, 640]
        self._input_h = int(shape[2]) if shape[2] != "?" else 640
        self._input_w = int(shape[3]) if shape[3] != "?" else 640

        out_shape = self.session.get_outputs()[0].shape
        # pose: 4 box + 1 conf + 17*3 kp = 56  |  detect: 4 + 80 = 84
        self._is_pose = (out_shape[1] == 4 + 1 + NUM_POSE_KP * 3)

        self.names = {i: n for i, n in enumerate(COCO_CLASSES)}

    def __call__(
        self,
        frame: np.ndarray,
        imgsz=None,   # accepted for API compatibility, ignored
        conf: float = 0.25,
        verbose: bool = False,
    ) -> list:
        target_hw = (self._input_h, self._input_w)
        img, ratio, pad_left, pad_top = _letterbox(frame, target_hw)

        # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis]  # add batch dim

        raw = self.session.run(None, {self._input_name: blob})[0]
        # raw shape: (1, channels, anchors) → transpose to (anchors, channels)
        preds = raw[0].T

        if self._is_pose:
            return self._parse_pose(preds, ratio, pad_left, pad_top, conf)
        return self._parse_detect(preds, ratio, pad_left, pad_top, conf)

    # ------------------------------------------------------------------
    def _unpad_boxes(self, xyxy, ratio, pad_left, pad_top):
        xyxy = xyxy.copy()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_left) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_top) / ratio
        return xyxy

    def _parse_detect(self, preds, ratio, pad_left, pad_top, conf_thr):
        # preds: (anchors, 4 + num_classes)
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(len(cls_ids)), cls_ids]

        mask = confs >= conf_thr
        if not mask.any():
            return [_Result()]

        boxes_xywh = boxes_xywh[mask]
        cls_ids = cls_ids[mask]
        confs = confs[mask]

        xyxy = self._unpad_boxes(_xywh_to_xyxy(boxes_xywh), ratio, pad_left, pad_top)
        keep = _nms(xyxy, confs)
        boxes = [_Box(xyxy[i], int(cls_ids[i]), float(confs[i])) for i in keep]
        return [_Result(boxes=_Boxes(boxes))]

    def _parse_pose(self, preds, ratio, pad_left, pad_top, conf_thr):
        # preds: (anchors, 4 + 1 + 17*3)
        boxes_xywh = preds[:, :4]
        confs = preds[:, 4]
        kp_flat = preds[:, 5:]  # (anchors, 51)

        mask = confs >= conf_thr
        if not mask.any():
            return [_Result()]

        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        kp_flat = kp_flat[mask]

        xyxy = self._unpad_boxes(_xywh_to_xyxy(boxes_xywh), ratio, pad_left, pad_top)
        keep = _nms(xyxy, confs)

        if not keep:
            return [_Result()]

        kp = kp_flat[keep].reshape(-1, NUM_POSE_KP, 3)  # (N, 17, 3)
        # undo letterbox on keypoint coordinates
        kp[:, :, 0] = (kp[:, :, 0] - pad_left) / ratio
        kp[:, :, 1] = (kp[:, :, 1] - pad_top) / ratio

        xy_only = kp[:, :, :2]  # (N, 17, 2) — drop visibility channel
        return [_Result(keypoints=_Keypoints(xy_only))]
