"""
Push-up repetition quality scoring system.
"""
import numpy as np


class RepScorer:
    """
    Analyzes and scores push-up repetition quality based on multiple metrics.

    Metrics evaluated:
    - ROM (Range of Motion): How deep and high the movement goes
    - Depth: Shoulder descent relative to hips
    - Body Line: Plank quality (shoulder-hip-ankle alignment)
    - Tempo: Speed of descent and ascent phases
    - Stability: Hip vertical movement consistency
    - Symmetry: Left/right elbow angle balance
    """

    def __init__(self):
        """Initialize scorer with empty frame buffer."""
        self.frames = []

    def reset(self):
        """Clear frame buffer for next repetition."""
        self.frames = []

    def add_frame(self, metrics: dict):
        """
        Record metrics for a single frame during repetition.

        Args:
            metrics: Dictionary containing:
                - elbowL: Left elbow angle (degrees)
                - elbowR: Right elbow angle (degrees)
                - shoulder_y: Normalized shoulder y-coordinate (0-1)
                - hip_y: Normalized hip y-coordinate (0-1)
                - line_dev: Body line deviation angle (degrees)
        """
        self.frames.append(metrics)

    def finalize(self, target_down_s=2.0, target_up_s=1.0, fps=30.0):
        """
        Calculate final score after repetition completes.

        Args:
            target_down_s: Target time for descent phase (seconds)
            target_up_s: Target time for ascent phase (seconds)
            fps: Frames per second for tempo calculation

        Returns:
            Dictionary with:
                - score: Overall score (0-100)
                - breakdown: Individual metric scores and details
                - notes: Any warnings or issues (if applicable)
        """
        if not self.frames:
            return {
                "score": 0.0,
                "breakdown": {},
                "notes": ["No frames captured"]
            }

        # Extract time series for each metric
        eL = np.array([m["elbowL"] for m in self.frames])
        eR = np.array([m["elbowR"] for m in self.frames])
        hip_y = np.array([m["hip_y"] for m in self.frames])
        sh_y = np.array([m["shoulder_y"] for m in self.frames])
        line_dev = np.array([m["line_dev"] for m in self.frames])

        # 1) Range of Motion (ROM) - 35% weight
        # Bottom: Lower angle is better (full flexion)
        rom_bottom_score = np.interp(min(eL.min(), eR.min()), [110, 70], [0, 100])
        # Top: Higher angle is better (full extension)
        rom_top_score = np.interp(max(eL.max(), eR.max()), [120, 160], [0, 100])
        ROM = 0.6 * rom_bottom_score + 0.4 * rom_top_score

        # 2) Depth - 15% weight
        # Vertical distance shoulder travels relative to hip
        depth_series = sh_y - hip_y
        depth_score = np.interp(depth_series.max(), [0.00, 0.08], [0, 100])

        # 3) Body Line (Plank Quality) - 20% weight
        # Lower deviation = straighter line = better form
        bodyline_score = np.interp(100 - np.percentile(line_dev, 90), [60, 100], [0, 100])

        # 4) Tempo - 10% weight
        # Find the lowest point (minimum angle)
        mean_elbow = (eL + eR) / 2.0
        min_idx = int(np.argmin(mean_elbow))

        # Calculate phase durations
        down_s = max(min_idx, 1) / fps
        up_s = max(len(self.frames) - 1 - min_idx, 1) / fps

        # Score based on deviation from target tempo
        tempo_down = 100 - min(abs(down_s - target_down_s) / max(1e-6, target_down_s), 1.0) * 100
        tempo_up = 100 - min(abs(up_s - target_up_s) / max(1e-6, target_up_s), 1.0) * 100
        tempo_score = 0.6 * tempo_down + 0.4 * tempo_up

        # 5) Stability - 10% weight
        # Lower hip movement variance = more stable
        stability_score = 100 - np.interp(np.std(hip_y), [0.00, 0.03], [0, 100])

        # 6) Symmetry - 10% weight
        # Similar left/right elbow angles at bottom and top
        sym_bottom = abs(eL[min_idx] - eR[min_idx])
        sym_top = abs(eL[-1] - eR[-1])
        symmetry_score = 100 - np.interp((sym_bottom + sym_top) / 2.0, [0.0, 15.0], [0, 100])

        # Compile breakdown
        breakdown = {
            "ROM": float(np.clip(ROM, 0, 100)),
            "Depth": float(np.clip(depth_score, 0, 100)),
            "BodyLine": float(np.clip(bodyline_score, 0, 100)),
            "Tempo": float(np.clip(tempo_score, 0, 100)),
            "Stability": float(np.clip(stability_score, 0, 100)),
            "Symmetry": float(np.clip(symmetry_score, 0, 100)),
            "down_seconds": float(round(down_s, 2)),
            "up_seconds": float(round(up_s, 2)),
        }

        # Calculate weighted final score
        final = (
                0.35 * breakdown["ROM"]
                + 0.15 * breakdown["Depth"]
                + 0.20 * breakdown["BodyLine"]
                + 0.10 * breakdown["Tempo"]
                + 0.10 * breakdown["Stability"]
                + 0.10 * breakdown["Symmetry"]
        )

        return {
            "score": float(round(final, 1)),
            "breakdown": breakdown
        }