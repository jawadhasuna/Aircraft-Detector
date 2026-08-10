"""
Core inference logic for the SAR aircraft detector.

Kept separate from modal_app.py so it can be tested locally without Modal:
    python detector.py path/to/image.jpg
"""

import io

import numpy as np
from PIL import Image

# Must match CLS_NAMES in the training notebook — order matters.
CLASS_NAMES = ["A220", "A320321", "A330", "ARJ21", "Boeing737", "Boeing787"]

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def letterbox(img: Image.Image, size: int = INPUT_SIZE):
    """
    Resize preserving aspect ratio, pad the rest with gray (114,114,114).

    This mirrors what Ultralytics does during training/validation. If you
    skip the padding and just squash the image to 640x640, boxes come out
    shifted, so this matters.
    """
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y, orig_w, orig_h


def preprocess(image_bytes: bytes):
    """bytes -> (NCHW float32 tensor, metadata needed to undo the letterbox)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    canvas, scale, pad_x, pad_y, orig_w, orig_h = letterbox(img)

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)                        # CHW
    tensor = np.expand_dims(arr, 0)                     # NCHW

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_w": orig_w,
        "orig_h": orig_h,
    }
    return np.ascontiguousarray(tensor), meta


def iou_batch(box, boxes):
    """IoU of one [x1,y1,x2,y2] box against an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    """Greedy non-maximum suppression. Returns indices to keep."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_batch(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]
    return keep


def postprocess(output, meta, conf_threshold=CONF_THRESHOLD):
    """
    output0 has shape [1, 10, 8400]:
      rows 0-3  = box as cx, cy, w, h (in 640x640 input pixel space)
      rows 4-9  = per-class confidence (sigmoid already applied by the
                  exported Detect head, so no extra activation needed)
    """
    pred = output[0]                    # [10, 8400]
    boxes_xywh = pred[:4, :]            # [4, 8400]
    class_scores = pred[4:, :]          # [6, 8400]

    best_class = class_scores.argmax(axis=0)
    best_score = class_scores.max(axis=0)

    mask = best_score >= conf_threshold
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[:, mask]
    best_class = best_class[mask]
    best_score = best_score[mask]

    cx, cy, w, h = boxes_xywh
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # undo letterbox: remove padding, then rescale to original image size
    x1 = (x1 - meta["pad_x"]) / meta["scale"]
    y1 = (y1 - meta["pad_y"]) / meta["scale"]
    x2 = (x2 - meta["pad_x"]) / meta["scale"]
    y2 = (y2 - meta["pad_y"]) / meta["scale"]

    x1 = np.clip(x1, 0, meta["orig_w"])
    y1 = np.clip(y1, 0, meta["orig_h"])
    x2 = np.clip(x2, 0, meta["orig_w"])
    y2 = np.clip(y2, 0, meta["orig_h"])

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class, so two different aircraft types can overlap
    detections = []
    for cls_id in np.unique(best_class):
        cls_mask = best_class == cls_id
        cls_boxes = boxes[cls_mask]
        cls_scores = best_score[cls_mask]
        for idx in nms(cls_boxes, cls_scores):
            detections.append(
                {
                    "box": [round(float(v), 1) for v in cls_boxes[idx]],
                    "score": round(float(cls_scores[idx]), 3),
                    "className": CLASS_NAMES[int(cls_id)],
                }
            )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def run_detection(session, image_bytes: bytes):
    """Full pipeline: bytes in, list of detections out."""
    tensor, meta = preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    detections = postprocess(output, meta)
    return detections, meta


if __name__ == "__main__":
    import sys
    import time

    import onnxruntime as ort

    model_path = "models/yolov8n_int8.onnx"
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if not image_path:
        print("Usage: python detector.py <image path>")
        sys.exit(1)

    print(f"Loading {model_path} ...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    with open(image_path, "rb") as f:
        data = f.read()

    t0 = time.time()
    dets, meta = run_detection(sess, data)
    elapsed = (time.time() - t0) * 1000

    print(f"Image: {meta['orig_w']}x{meta['orig_h']}  |  {elapsed:.0f} ms")
    print(f"Detections: {len(dets)}")
    for d in dets:
        print(f"  {d['className']:<12} {d['score']:.3f}  box={d['box']}")
