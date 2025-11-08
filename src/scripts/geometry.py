"""
Geometric calculation utilities for pose analysis.
"""
import numpy as np


def angle(a, b, c):
    """
    Calculate the angle ABC (at point b) in degrees.

    Args:
        a, b, c: Points as array-like (x, y) coordinates

    Returns:
        Angle in degrees (0-180)
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)

    if nba < 1e-9 or nbc < 1e-9:
        return 0.0

    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))  # Clamp to valid range

    return float(np.degrees(np.arccos(cosv)))


def collinearity(a, b, c):
    """
    Measure collinearity of three points (how close they are to a straight line).

    Returns the angle at point b - smaller angles indicate better alignment.
    Useful for checking plank quality (shoulder-hip-ankle alignment).

    Args:
        a, b, c: Points as array-like (x, y) coordinates

    Returns:
        Deviation angle in degrees (lower is more collinear)
    """
    return angle(a, b, c)


def arm_torso_angle_diff(shoulder, elbow, hip):
    """
    Calculate the angle difference between upper arm and torso.

    Used to check if upper arm is parallel to the back during push-ups.
    When this difference is small (<20°), the upper arm is parallel to torso.

    Args:
        shoulder: Shoulder position (x, y)
        elbow: Elbow position (x, y)
        hip: Hip position (x, y)

    Returns:
        Absolute angle difference in degrees (lower means more parallel)
    """
    shoulder = np.array(shoulder, dtype=float)
    elbow = np.array(elbow, dtype=float)
    hip = np.array(hip, dtype=float)

    # Calculate angle of upper arm relative to horizontal
    arm_vec = elbow - shoulder
    arm_angle = float(np.degrees(np.arctan2(arm_vec[1], arm_vec[0])))

    # Calculate angle of torso relative to horizontal
    torso_vec = hip - shoulder
    torso_angle = float(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))

    # Return absolute difference
    diff = abs(arm_angle - torso_angle)

    # Normalize to [0, 180] range
    if diff > 180:
        diff = 360 - diff

    return diff


def is_in_pushup_position(shoulder, hip, knee, ankle, nose, min_plank_angle=160,
                          max_body_tilt=30, face_down_threshold=0.3):
    """
    Validate that the person is in a proper push-up plank position.

    Checks:
    1. Legs are straight (knee angle close to 180°)
    2. Body is horizontal (not standing/sitting)
    3. Face is pointing down (nose below shoulders)
    4. Body alignment is good (shoulder-hip-ankle relatively straight)

    Args:
        shoulder: Shoulder midpoint (x, y)
        hip: Hip midpoint (x, y)
        knee: Knee midpoint (x, y)
        ankle: Ankle midpoint (x, y)
        nose: Nose position (x, y)
        min_plank_angle: Minimum knee angle for straight legs (default 160°)
        max_body_tilt: Maximum tilt from horizontal for body (default 30°)
        face_down_threshold: Minimum y-distance ratio for face-down check (default 0.3)

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    shoulder = np.array(shoulder, dtype=float)
    hip = np.array(hip, dtype=float)
    knee = np.array(knee, dtype=float)
    ankle = np.array(ankle, dtype=float)
    nose = np.array(nose, dtype=float)

    # Check 1: Legs should be straight (knee angle ~180°)
    knee_angle = angle(hip, knee, ankle)
    if knee_angle < min_plank_angle:
        return False, f"Legs bent ({int(knee_angle)}° < {min_plank_angle}°)"

    # Check 2: Body should be roughly horizontal (torso angle)
    torso_vec = hip - shoulder
    torso_angle_from_horizontal = abs(np.degrees(np.arctan2(torso_vec[1], torso_vec[0])))
    if torso_angle_from_horizontal > max_body_tilt:
        return False, f"Not horizontal ({int(torso_angle_from_horizontal)}° > {max_body_tilt}°)"

    # Check 3: Face should be pointing down (nose below shoulders in y-axis)
    # In image coordinates, y increases downward, so nose.y should be > shoulder.y
    body_height = abs(shoulder[1] - ankle[1])
    face_down_dist = nose[1] - shoulder[1]

    if body_height > 0:
        face_ratio = face_down_dist / body_height
        if face_ratio < face_down_threshold:
            return False, "Face not pointing down"

    return True, "Valid push-up position"