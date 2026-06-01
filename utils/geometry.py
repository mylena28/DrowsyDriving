import numpy as np

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalized_distance(p1, p2, ref1, ref2):
    d = distance(p1, p2)
    d_ref = distance(ref1, ref2)

    if d_ref == 0:
        return 0

    return d / d_ref

def lower_face_crop(frame, nose, eye_l, eye_r):
    """Crop the lower-face region (nose down to chin) for small-object detection.

    Uses inter-eye distance as the face scale. Falls back to the full frame if
    any keypoint is missing or the face is too small to be reliable.
    """
    if nose is None or eye_l is None or eye_r is None:
        return frame

    eye_dist = distance(eye_l, eye_r)
    if eye_dist < 10:
        return frame

    cx, cy = int(nose[0]), int(nose[1])
    h, w = frame.shape[:2]

    # Horizontal: 1.2× eye-distance either side of nose
    # Vertical: 0.5× above nose (to include nose tip context), 1.8× below (chin)
    margin_x    = int(eye_dist * 1.2)
    margin_up   = int(eye_dist * 0.5)
    margin_down = int(eye_dist * 1.8)

    x1 = max(0, cx - margin_x)
    x2 = min(w, cx + margin_x)
    y1 = max(0, cy - margin_up)
    y2 = min(h, cy + margin_down)

    if x2 <= x1 or y2 <= y1:
        return frame

    return frame[y1:y2, x1:x2]
