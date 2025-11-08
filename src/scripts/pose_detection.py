"""
Pose detection using MoveNet and keypoint utilities.
"""
import cv2
import numpy as np

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except Exception as e:
    raise SystemExit(
        "Failed to import TensorFlow / TF Hub. Install with:\n"
        "  pip install 'tensorflow<2.17' tensorflow-hub opencv-python\n\n"
        f"Original error: {e}"
    )

# MoveNet SinglePose keypoint order
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

NAME_TO_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def load_movenet(model="thunder"):
    """
    Load MoveNet SinglePose model from TensorFlow Hub.

    Args:
        model: Either 'thunder' (more accurate, 256x256) or
               'lightning' (faster, 192x192)

    Returns:
        Tuple of (model_function, input_size)
    """
    if model == "thunder":
        handle = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        input_size = 256
    else:
        handle = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        input_size = 192

    try:
        m = hub.load(handle)
        return m.signatures["serving_default"], input_size
    except Exception as e:
        raise SystemExit(
            "Failed to load MoveNet from TF Hub.\n"
            "Check your internet and ensure tensorflow-hub is installed.\n"
            f"Original error: {e}"
        )


def draw_skeleton(frame, kps_xy, conf, min_conf=0.3):
    """
    Draw skeleton overlay on frame.

    Args:
        frame: OpenCV frame (modified in-place)
        kps_xy: Array of keypoint (x, y) pixel coordinates, shape (17, 2)
        conf: Array of keypoint confidence scores, shape (17,)
        min_conf: Minimum confidence threshold to draw keypoint/connection
    """
    # Define skeleton connections
    pairs = [
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]

    # Draw connections
    for a, b in pairs:
        ia, ib = NAME_TO_IDX[a], NAME_TO_IDX[b]
        if conf[ia] >= min_conf and conf[ib] >= min_conf:
            pa = tuple(map(int, kps_xy[ia]))
            pb = tuple(map(int, kps_xy[ib]))
            cv2.line(frame, pa, pb, (255, 255, 255), 2)

    # Draw keypoints
    for i, p in enumerate(kps_xy):
        if conf[i] >= min_conf:
            cv2.circle(frame, tuple(map(int, p)), 3, (0, 255, 0), -1)


def get_keypoint(name, keypoints_xy, confidences):
    """
    Get a specific keypoint by name.

    Args:
        name: Keypoint name (e.g., "left_shoulder")
        keypoints_xy: Array of (x, y) coordinates, shape (17, 2)
        confidences: Array of confidence scores, shape (17,)

    Returns:
        Tuple of (position, confidence) where position is (x, y) array
    """
    idx = NAME_TO_IDX[name]
    return keypoints_xy[idx], float(confidences[idx])


def preprocess_frame(frame, input_size):
    """
    Preprocess frame for MoveNet inference.

    Args:
        frame: OpenCV BGR frame
        input_size: Model input size (e.g., 256 for Thunder)

    Returns:
        TensorFlow tensor ready for model input
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_size, input_size))
    input_tensor = tf.convert_to_tensor(img_resized, dtype=tf.int32)
    input_tensor = tf.expand_dims(input_tensor, axis=0)
    return input_tensor


def extract_keypoints(model_output):
    """
    Extract keypoints from MoveNet output.

    Args:
        model_output: Dictionary from model inference

    Returns:
        Tuple of (keypoints_yx, confidences) where:
            - keypoints_yx: (17, 2) array of normalized (y, x) coordinates
            - confidences: (17,) array of confidence scores
    """
    # MoveNet output: [1, 1, 17, 3] with format (y, x, score)
    kps = model_output["output_0"].numpy()[0, 0, :, :]
    keypoints_yx = kps[:, :2]  # (y, x) in normalized coords [0, 1]
    confidences = kps[:, 2]
    return keypoints_yx, confidences