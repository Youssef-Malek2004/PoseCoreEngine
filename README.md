# Push-up Counter with Form Analysis

Real-time push-up counter and form analyzer using webcam, MoveNet pose detection, and quality scoring.

## Project Structure

```
.
├── main.py              # Main application entry point
├──scripts
├──── filters.py           # One Euro Filter for smoothing
├──── geometry.py          # Geometric calculations (angles, collinearity)
├──── counter.py           # Rep counting state machine
├──── scorer.py            # Rep quality scoring system
├──── pose_detection.py    # MoveNet model and keypoint utilities
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Module Overview

### `filters.py`
- **OneEuroFilter**: Adaptive low-pass filter for reducing jitter in time-series data
- **OneEuro2D**: Wrapper for filtering 2D points (x, y coordinates)
- Based on "The One Euro Filter" (Casiez et al., 2012)

### `geometry.py`
- **angle()**: Calculate angle between three points
- **collinearity()**: Measure how close three points are to a straight line
- Used for analyzing body alignment and joint angles

### `counter.py`
- **PushupCounter**: State machine for counting repetitions with proper form
- Counts reps when:
  1. Start from extended position (elbow > 140°)
  2. Reach 90° elbow angle with upper arm parallel to back
  3. Return to extended position (elbow > 140°)
- Includes frame buffers to avoid false counts from jittery movements
- Validates proper push-up form by checking arm-to-back parallelism

### `scorer.py`
- **RepScorer**: Analyzes and scores push-up quality (0-100)
- Evaluates 6 metrics:
  - **ROM** (35%): Range of motion - depth and extension
  - **Depth** (15%): Shoulder descent relative to hips  
  - **Body Line** (20%): Plank quality (alignment)
  - **Tempo** (10%): Speed of descent/ascent phases
  - **Stability** (10%): Hip movement consistency
  - **Symmetry** (10%): Left/right balance

### `pose_detection.py`
- **load_movenet()**: Load MoveNet model from TensorFlow Hub
- **draw_skeleton()**: Visualize detected keypoints on frame
- **get_keypoint()**: Extract specific keypoint by name
- **preprocess_frame()**: Prepare frame for model inference
- **extract_keypoints()**: Parse model output

### `main.py`
- Integrates all modules into a real-time application
- Handles webcam input, visualization, and user interface
- Displays rep count, current score, and FPS

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage with default settings:
```bash
python main.py
```

### Command-line Options

```bash
python main.py [OPTIONS]

Options:
  --model {lightning,thunder}  MoveNet variant (default: thunder)
                               thunder = more accurate, 256x256 input
                               lightning = faster, 192x192 input
  
  --device INT                 Webcam device index (default: 0)
  
  --min_conf FLOAT            Minimum keypoint confidence (default: 0.3)
  
  --freq FLOAT                Expected camera FPS for filter (default: 60.0)
  
  --min_cutoff FLOAT          One Euro min cutoff frequency (default: 1.0)
  
  --beta FLOAT                One Euro beta/speed coefficient (default: 0.1)
  
  --d_cutoff FLOAT            One Euro derivative cutoff (default: 1.0)
  
  --mirror                    Mirror camera horizontally (selfie mode)
  
  --show_debug                Show debug metrics on screen
```

### Examples

Fast mode with lighter model:
```bash
python main.py --model lightning
```

Selfie mode with mirrored camera:
```bash
python main.py --mirror
```

Adjust smoothing (less smooth, more responsive):
```bash
python main.py --beta 0.5 --min_cutoff 2.0
```

## How It Works

1. **Capture Frame**: Read from webcam
2. **Pose Detection**: Run MoveNet to detect 17 body keypoints
3. **Smoothing**: Apply One Euro filter to reduce jitter
4. **Analysis**: Calculate joint angles and body alignment
5. **Form Validation**: Check if upper arm is parallel to back at 90° elbow
6. **Counting**: Update state machine to count reps (only counts proper form)
7. **Scoring**: Evaluate rep quality across 6 metrics
8. **Display**: Show results with visual feedback

## Scoring Breakdown

Each rep is scored 0-100 based on:

| Metric | Weight | Description |
|--------|--------|-------------|
| ROM | 35% | How deep you go (elbow flexion) and full extension |
| Body Line | 20% | Plank quality - straight line from shoulders to ankles |
| Depth | 15% | How much your shoulders descend relative to hips |
| Tempo | 10% | Maintaining proper speed (2s down, 1s up) |
| Stability | 10% | Keeping hips stable without sagging |
| Symmetry | 10% | Equal left/right elbow angles |

## Requirements

- Python 3.7+
- Webcam
- TensorFlow < 2.17
- OpenCV
- NumPy

## Controls

- **ESC** or **Q**: Quit application

## Tips for Best Results

1. **Lighting**: Ensure good, even lighting on your body
2. **Camera Position**: Place camera to capture full body in side view
3. **Distance**: Stand far enough that shoulders, hips, and ankles are visible
4. **Clothing**: Wear fitted clothing for better pose detection
5. **Background**: Plain background works best

## Technical Notes

### One Euro Filter Parameters

- **freq**: Expected update rate (Hz) - should match your camera FPS
- **min_cutoff**: Lower = smoother but more lag (1.0 is balanced)
- **beta**: Speed coefficient - higher = more responsive to fast movements (0.1 is conservative)
- **d_cutoff**: Derivative filter cutoff (1.0 is typical)

### Rep Counter Thresholds

Default values (balanced for proper form):
- **down_angle**: 90° - target elbow angle for "down" position
- **angle_tolerance**: 15° - accepts 75-105° range
- **up_th**: 140° - elbow angle to register "up" position and extension
- **parallel_th**: 20° - max angle difference for arm-back parallelism
- **min_down_frames**: 3 - frames needed in down position
- **min_up_frames**: 3 - frames needed in up position to count

### What is "Parallel to Back"?

The counter validates proper push-up form by checking if your upper arm (shoulder to elbow) is parallel to your torso (shoulder to hip) when at the 90° position. This ensures you're:
- Keeping elbows close to body (not flaring out)
- Maintaining proper form throughout the movement
- Getting full range of motion at the bottom

When the arm-to-back angle difference is ≤20°, you'll see **[PARALLEL]** in green on screen.

### Adjusting Strictness

You can modify the counter strictness in `main.py`:

**More Lenient** (easier to count, less strict on form):
```python
counter = PushupCounter(
    down_angle=90,
    angle_tolerance=20,      # Accept wider range (70-110°)
    up_th=130,              # Lower extension requirement
    parallel_th=30,         # More lenient parallelism (up to 30° diff)
    min_down_frames=2,
    min_up_frames=2
)
```

**Current (Balanced - Proper Form)**:
```python
counter = PushupCounter(
    down_angle=90,
    angle_tolerance=15,      # 75-105° range
    up_th=140,              # Good extension
    parallel_th=20,         # Strict parallelism
    min_down_frames=3,
    min_up_frames=3
)
```

**Very Strict (Competition/Military Standard)**:
```python
counter = PushupCounter(
    down_angle=90,
    angle_tolerance=10,      # 80-100° range
    up_th=160,              # Near-full extension
    parallel_th=15,         # Very strict parallelism
    min_down_frames=5,
    min_up_frames=5
)
```

**Lenient Mode (No Parallelism Check)**:
To disable the parallelism requirement entirely, modify line 219 in `main.py`:
```python
# Pass None instead of arm_back_diff
rep_completed = counter.update(elbow_avg, None)
```

These can be adjusted in `main.py` to match different exercise styles or fitness levels.

## Future Enhancements

Potential improvements:
- Support for other exercises (squats, planks, etc.)
- Real-time audio feedback
- Historical tracking and progress charts
- Multi-person detection
- Mobile app version
- Cloud-based analysis and coaching

## License

This project is for educational and personal use.

## Credits

- MoveNet pose detection by Google Research
- One Euro Filter by Géry Casiez, Nicolas Roussel, and Daniel Vogel