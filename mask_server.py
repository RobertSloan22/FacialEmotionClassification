"""
Face Masking Server
-------------------
Flask backend that receives a base64-encoded video frame and returns the
same frame with the detected face(s) masked according to the chosen style.

Mask types
----------
blur       – Gaussian blur over the face region
pixelate   – Pixelate the face region (block effect)
blackbox   – Solid black rectangle over the face
whitebox   – Solid white rectangle over the face
oval_blur  – Elliptical Gaussian blur (follows face shape)
none       – No masking (pass-through, useful for testing)

Run with:  python mask_server.py
Default port: 5001
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)
CORS(app)

_face_detection = mp.solutions.face_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(b64_str: str) -> np.ndarray:
    """Decode a base64 data-URI or raw base64 string into a BGR numpy array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _encode_image(img: np.ndarray) -> str:
    """Encode a BGR numpy array to a JPEG base64 data-URI."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def _detect_faces(img: np.ndarray):
    """Return a list of (x, y, w, h) bounding boxes for each detected face."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with _face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as detector:
        results = detector.process(rgb)

    boxes = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            # Convert relative coords to absolute, add 10 % padding
            bx = int(bb.xmin * w)
            by = int(bb.ymin * h)
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            pad = int(min(bw, bh) * 0.10)
            bx = max(0, bx - pad)
            by = max(0, by - pad)
            bw = min(bw + 2 * pad, w - bx)
            bh = min(bh + 2 * pad, h - by)
            boxes.append((bx, by, bw, bh))
    return boxes


# ---------------------------------------------------------------------------
# Mask implementations
# ---------------------------------------------------------------------------

def _blur(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        roi = out[y : y + h, x : x + w]
        # Kernel size must be odd and > 0
        ksize = max(31, (min(w, h) // 2) | 1)
        out[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
    return out


def _pixelate(img: np.ndarray, boxes, block: int = 16) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        roi = out[y : y + h, x : x + w]
        small_w = max(1, w // block)
        small_h = max(1, h // block)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        out[y : y + h, x : x + w] = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_NEAREST
        )
    return out


def _blackbox(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        out[y : y + h, x : x + w] = 0
    return out


def _whitebox(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        out[y : y + h, x : x + w] = 255
    return out


def _oval_blur(img: np.ndarray, boxes) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        roi = img[y : y + h, x : x + w].copy()
        ksize = max(31, (min(w, h) // 2) | 1)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # Elliptical mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        mask3 = cv2.merge([mask, mask, mask])
        inv3 = cv2.bitwise_not(mask3)

        out[y : y + h, x : x + w] = cv2.add(
            cv2.bitwise_and(blurred, mask3), cv2.bitwise_and(roi, inv3)
        )
    return out


_MASK_FNS = {
    "blur": _blur,
    "pixelate": _pixelate,
    "blackbox": _blackbox,
    "whitebox": _whitebox,
    "oval_blur": _oval_blur,
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/mask", methods=["POST"])
def mask_face():
    data = request.get_json(force=True)
    b64 = data.get("image", "")
    mask_type = data.get("mask_type", "blur")

    if not b64:
        return jsonify({"error": "No image provided"}), 400

    img = _decode_image(b64)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    boxes = _detect_faces(img)

    if mask_type == "none" or not boxes:
        result = img
    else:
        fn = _MASK_FNS.get(mask_type, _blur)
        result = fn(img, boxes)

    return jsonify(
        {"image": _encode_image(result), "faces_detected": len(boxes)}
    )


if __name__ == "__main__":
    print("Face Masking Server starting on http://localhost:5001")
    print("Mask types: blur | pixelate | blackbox | whitebox | oval_blur | none")
    app.run(host="0.0.0.0", port=5001, debug=False)
