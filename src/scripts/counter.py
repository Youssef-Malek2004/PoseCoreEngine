"""
Push-up repetition counter with state machine logic.
"""


class PushupCounter:
    """
    State machine for counting push-up repetitions based on elbow angle and arm alignment.

    Tracks "up" and "down" states. A rep is counted when:
    1. Start from extended position (elbow > up_th)
    2. Reach 90° elbow with upper arm parallel to back (down position)
    3. Return to extended position (elbow > up_th)
    """

    def __init__(self, down_angle=90, angle_tolerance=15, up_th=140,
                 parallel_th=20, min_down_frames=3, min_up_frames=3):
        """
        Initialize counter.

        Args:
            down_angle: Target elbow angle for "down" position (degrees, typically 90)
            angle_tolerance: Tolerance around down_angle (degrees, typically 15)
            up_th: Elbow angle threshold for "up" position (degrees, typically 140-160)
            parallel_th: Max angle difference for arm-to-back parallelism (degrees, typically 20)
            min_down_frames: Minimum frames in down position to register state
            min_up_frames: Minimum frames in up position to register state and count rep
        """
        self.state = "up"
        self.reps = 0
        self.down_frames = 0
        self.up_frames = 0
        self.down_angle = down_angle
        self.angle_tolerance = angle_tolerance
        self.up_th = up_th
        self.parallel_th = parallel_th
        self.min_down_frames = min_down_frames
        self.min_up_frames = min_up_frames

    def update(self, elbow_angle, arm_back_angle_diff=None):
        """
        Update state machine with new measurements.

        Args:
            elbow_angle: Current average elbow angle in degrees
            arm_back_angle_diff: Angle difference between upper arm and back (degrees).
                                 If None, parallelism check is skipped (lenient mode).

        Returns:
            True if a repetition was just completed, False otherwise
        """
        rep_completed = False

        # Check if in valid down position (90° ± tolerance AND parallel to back)
        in_down_angle_range = abs(elbow_angle - self.down_angle) <= self.angle_tolerance

        if arm_back_angle_diff is not None:
            # Strict mode: require both angle and parallelism
            is_parallel = arm_back_angle_diff <= self.parallel_th
            in_down_position = in_down_angle_range and is_parallel
        else:
            # Lenient mode: only check angle
            in_down_position = in_down_angle_range

        # State machine logic
        if in_down_position:
            # In down position (90° + parallel)
            self.down_frames += 1
            self.up_frames = 0
            if self.state == "up" and self.down_frames >= self.min_down_frames:
                self.state = "down"

        elif elbow_angle > self.up_th:
            # In up position (extended)
            self.up_frames += 1
            self.down_frames = 0
            if self.state == "down" and self.up_frames >= self.min_up_frames:
                self.state = "up"
                self.reps += 1
                rep_completed = True

        else:
            # In transition zone - decay frame counters
            self.down_frames = max(0, self.down_frames - 1)
            self.up_frames = max(0, self.up_frames - 1)

        return rep_completed

    def reset(self):
        """Reset counter to initial state."""
        self.state = "up"
        self.reps = 0
        self.down_frames = 0
        self.up_frames = 0