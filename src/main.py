"""
Real-time push-up counter and form analyzer using webcam + MoveNet.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from scripts.filters import OneEuro2D
from scripts.geometry import angle, collinearity, arm_torso_angle_diff, is_in_pushup_position
from scripts.counter import PushupCounter
from scripts.scorer import RepScorer
from scripts.pose_detection import (
    load_movenet,
    draw_skeleton,
    get_keypoint,
    preprocess_frame,
    extract_keypoints,
    NAME_TO_IDX
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Push-up counter with MoveNet + One Euro filtering"
    )
    parser.add_argument(
        "--model",
        choices=["lightning", "thunder"],
        default="thunder",
        help="MoveNet model variant (thunder=accurate, lightning=fast)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index"
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.3,
        help="Minimum keypoint confidence threshold"
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=60.0,
        help="Expected camera FPS for One Euro filter"
    )
    parser.add_argument(
        "--min_cutoff",
        type=float,
        default=1.0,
        help="One Euro filter minimum cutoff frequency"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="One Euro filter beta (speed coefficient)"
    )
    parser.add_argument(
        "--d_cutoff",
        type=float,
        default=1.0,
        help="One Euro filter derivative cutoff"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror camera horizontally (selfie mode)"
    )
    parser.add_argument(
        "--show_debug",
        action="store_true",
        help="Show debug metrics on screen"
    )
    parser.add_argument(
        "--lenient_position",
        action="store_true",
        help="Disable strict push-up position validation"
    )
    args = parser.parse_args()

    # Load MoveNet model
    print(f"Loading MoveNet {args.model} model...")
    movenet, input_size = load_movenet(args.model)
    print("Model loaded successfully!")

    # Open webcam
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam device {args.device}")

    # Initialize One Euro filters for each keypoint (17 keypoints, 2D each)
    filters = [
        OneEuro2D(
            freq=args.freq,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            d_cutoff=args.d_cutoff
        )
        for _ in range(17)
    ]

    # Initialize counter and scorer
    counter = PushupCounter(
        down_angle=80,        # Target 90° elbow angle for down position
        angle_tolerance=30,   # Accept 50-110° range
        up_th=120,           # Must extend to >120° for up position
        parallel_th=40,      # Max 40° difference for arm-back parallelism
        min_down_frames=1,   # Frames needed in down position
        min_up_frames=1      # Frames needed in up position to count rep
    )
    rep_scorer = RepScorer()
    in_rep = False

    # FPS tracking
    fps_hist = deque(maxlen=30)
    prev_time = time.time()

    # Position validation tracking
    position_warning_frames = 0
    POSITION_WARNING_COOLDOWN = 90  # Show warning for 3 seconds at 30fps

    print("\nStarting real-time push-up counter...")
    print("Press 'q' or ESC to quit\n")
    if not args.lenient_position:
        print("⚠️  Push-up position validation ENABLED")
        print("   Make sure you're in plank position with:")
        print("   - Legs straight")
        print("   - Body horizontal")
        print("   - Face pointing down\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # Preprocess and run pose detection
        input_tensor = preprocess_frame(frame, input_size)
        outputs = movenet(input_tensor)
        kps_yx, kps_conf = extract_keypoints(outputs)

        # Apply One Euro filter to smooth keypoints
        t_now = time.time()
        filtered_xy = []
        for i in range(17):
            y, x = float(kps_yx[i, 0]), float(kps_yx[i, 1])
            # Filter in normalized space, then map to pixels
            xy_f = filters[i]([x, y], t=t_now)
            px = np.array([xy_f[0] * w, xy_f[1] * h])
            filtered_xy.append(px)
        filtered_xy = np.array(filtered_xy, dtype=float)

        # Draw skeleton on frame
        draw_skeleton(frame, filtered_xy, kps_conf, min_conf=args.min_conf)

        # Extract required keypoints for analysis
        needed_keypoints = [
            "nose",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

        # Check if all needed keypoints are visible
        all_visible = all(
            kps_conf[NAME_TO_IDX[name]] >= args.min_conf
            for name in needed_keypoints
        )

        if all_visible:
            # Get keypoint positions
            nose, _ = get_keypoint("nose", filtered_xy, kps_conf)
            ls, _ = get_keypoint("left_shoulder", filtered_xy, kps_conf)
            rs, _ = get_keypoint("right_shoulder", filtered_xy, kps_conf)
            le, _ = get_keypoint("left_elbow", filtered_xy, kps_conf)
            re, _ = get_keypoint("right_elbow", filtered_xy, kps_conf)
            lw, _ = get_keypoint("left_wrist", filtered_xy, kps_conf)
            rw, _ = get_keypoint("right_wrist", filtered_xy, kps_conf)
            lh, _ = get_keypoint("left_hip", filtered_xy, kps_conf)
            rh, _ = get_keypoint("right_hip", filtered_xy, kps_conf)
            lk, _ = get_keypoint("left_knee", filtered_xy, kps_conf)
            rk, _ = get_keypoint("right_knee", filtered_xy, kps_conf)
            la, _ = get_keypoint("left_ankle", filtered_xy, kps_conf)
            ra, _ = get_keypoint("right_ankle", filtered_xy, kps_conf)

            # Calculate midpoints
            shoulder = (ls + rs) / 2.0
            hip = (lh + rh) / 2.0
            knee = (lk + rk) / 2.0
            ankle = (la + ra) / 2.0

            # Validate push-up position (unless disabled)
            is_valid_position = True
            position_reason = ""

            if not args.lenient_position:
                is_valid_position, position_reason = is_in_pushup_position(
                    shoulder, hip, knee, ankle, nose,
                    min_plank_angle=160,      # Legs should be quite straight
                    max_body_tilt=30,         # Body shouldn't be too tilted
                    face_down_threshold=0.2   # Face should be pointing down
                )

                if not is_valid_position:
                    position_warning_frames = POSITION_WARNING_COOLDOWN

            # Calculate elbow angles
            elbowL = angle(ls, le, lw)
            elbowR = angle(rs, re, rw)
            elbow_avg = (elbowL + elbowR) / 2.0

            # Calculate body line deviation (plank quality)
            line_dev = collinearity(shoulder, hip, ankle)

            # Calculate arm-to-back parallelism (for proper push-up form)
            # Average the left and right sides
            arm_back_diff_L = arm_torso_angle_diff(ls, le, lh)
            arm_back_diff_R = arm_torso_angle_diff(rs, re, rh)
            arm_back_diff = (arm_back_diff_L + arm_back_diff_R) / 2.0

            # Normalize y-coordinates for consistent scoring
            shoulder_y_norm = shoulder[1] / max(1.0, h)
            hip_y_norm = hip[1] / max(1.0, h)

            # Track rep phases (only if in valid position)
            if is_valid_position:
                if not in_rep and elbow_avg < 160:  # Starting a rep
                    in_rep = True
                    rep_scorer.reset()

                if in_rep:
                    rep_scorer.add_frame({
                        "elbowL": elbowL,
                        "elbowR": elbowR,
                        "shoulder_y": shoulder_y_norm,
                        "hip_y": hip_y_norm,
                        "line_dev": line_dev
                    })

                # Update counter with elbow angle and arm-back parallelism
                rep_completed = counter.update(elbow_avg, arm_back_diff)

                # Score and display completed rep
                if rep_completed and in_rep:
                    avg_fps = np.mean(fps_hist) if fps_hist else 30.0
                    result = rep_scorer.finalize(fps=avg_fps)
                    in_rep = False

                    # Display score overlay
                    score = result["score"]
                    cv2.rectangle(frame, (10, 60), (320, 135), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        f"Rep {counter.reps} score: {score:.1f}/100",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    # Print detailed breakdown to console
                    print(f"\n[Rep {counter.reps}] Score: {score:.1f}/100")
                    print(f"  Breakdown: {result['breakdown']}")

            # Display position warning if not valid
            if position_warning_frames > 0:
                warning_alpha = min(1.0, position_warning_frames / 30.0)
                cv2.rectangle(frame, (10, h - 150), (w - 10, h - 80), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    "WARNING: NOT IN PUSH-UP POSITION",
                    (20, h - 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )
                cv2.putText(
                    frame,
                    position_reason,
                    (20, h - 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                position_warning_frames -= 1

            # Display current metrics
            is_parallel = arm_back_diff <= counter.parallel_th
            parallel_color = (0, 255, 0) if is_parallel else (0, 165, 255)

            # Position indicator
            position_color = (0, 255, 0) if is_valid_position else (0, 0, 255)
            position_text = "READY" if is_valid_position else "INVALID POS"

            cv2.putText(
                frame,
                position_text,
                (w - 180, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                position_color,
                2
            )

            cv2.putText(
                frame,
                f"Elbow: {int(elbow_avg)} deg",
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Arm-Back: {int(arm_back_diff)} deg {'[PARALLEL]' if is_parallel else ''}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                parallel_color,
                2
            )

        # Calculate and track FPS
        now = time.time()
        fps = 1.0 / max(1e-6, now - prev_time)
        prev_time = now
        fps_hist.append(fps)

        # Draw main UI overlays
        cv2.rectangle(frame, (8, 8), (240, 55), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Reps: {counter.reps}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"FPS: {np.mean(fps_hist):.1f}",
            (w - 140, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display frame
        cv2.imshow("MoveNet Push-up Counter", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession complete! Total reps: {counter.reps}")


if __name__ == "__main__":
    main()