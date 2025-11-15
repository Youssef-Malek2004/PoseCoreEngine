// Push-up Counter Logic
class PushupCounter {
    constructor() {
        this.state = 'up';
        this.reps = 0;
        this.downFrames = 0;
        this.upFrames = 0;
        this.downAngle = 80;
        this.angleTolerance = 30;
        this.upThreshold = 120;
        this.parallelThreshold = 40;
        this.minDownFrames = 1;
        this.minUpFrames = 1;
    }

    update(elbowAngle, armBackDiff) {
        let repCompleted = false;

        // Check if in down position
        const inDownAngleRange = Math.abs(elbowAngle - this.downAngle) <= this.angleTolerance;
        const isParallel = armBackDiff !== null && armBackDiff <= this.parallelThreshold;
        const inDownPosition = armBackDiff === null ? inDownAngleRange : (inDownAngleRange && isParallel);

        if (inDownPosition) {
            this.downFrames++;
            this.upFrames = 0;
            if (this.state === 'up' && this.downFrames >= this.minDownFrames) {
                this.state = 'down';
            }
        } else if (elbowAngle > this.upThreshold) {
            this.upFrames++;
            this.downFrames = 0;
            if (this.state === 'down' && this.upFrames >= this.minUpFrames) {
                this.state = 'up';
                this.reps++;
                repCompleted = true;
            }
        } else {
            this.downFrames = Math.max(0, this.downFrames - 1);
            this.upFrames = Math.max(0, this.upFrames - 1);
        }

        return repCompleted;
    }

    reset() {
        this.reps = 0;
        this.state = 'up';
        this.downFrames = 0;
        this.upFrames = 0;
    }
}

// Geometry utilities
function calculateAngle(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y };
    const bc = { x: c.x - b.x, y: c.y - b.y };

    const dotProduct = ba.x * bc.x + ba.y * bc.y;
    const magnitudeBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magnitudeBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);

    if (magnitudeBA < 1e-9 || magnitudeBC < 1e-9) return 0;

    const cosAngle = dotProduct / (magnitudeBA * magnitudeBC);
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));

    return Math.acos(clampedCos) * (180 / Math.PI);
}

function armTorsoAngleDiff(shoulder, elbow, hip) {
    const armVec = { x: elbow.x - shoulder.x, y: elbow.y - shoulder.y };
    const torsoVec = { x: hip.x - shoulder.x, y: hip.y - shoulder.y };

    const armAngle = Math.atan2(armVec.y, armVec.x) * (180 / Math.PI);
    const torsoAngle = Math.atan2(torsoVec.y, torsoVec.x) * (180 / Math.PI);

    let diff = Math.abs(armAngle - torsoAngle);
    if (diff > 180) diff = 360 - diff;

    return diff;
}

function isInPushupPosition(shoulder, hip, knee, ankle, nose) {
    // Check 1: Legs straight (knee angle)
    const kneeAngle = calculateAngle(hip, knee, ankle);
    if (kneeAngle < 120) {
        return { valid: false, reason: `Legs bent (${Math.round(kneeAngle)}° < 120°)` };
    }

    // Check 2: Body horizontal
    const torsoVec = { x: hip.x - shoulder.x, y: hip.y - shoulder.y };
    const torsoAngleFromHorizontal = Math.abs(Math.atan2(torsoVec.y, torsoVec.x) * (180 / Math.PI));
    if (torsoAngleFromHorizontal > 50) {
        return { valid: false, reason: `Not horizontal (${Math.round(torsoAngleFromHorizontal)}° > 50°)` };
    }

    // Check 3: Face pointing down (nose below shoulders in y-axis)
    const bodyHeight = Math.abs(shoulder.y - ankle.y);
    const faceDownDist = nose.y - shoulder.y;

    if (bodyHeight > 0) {
        const faceRatio = faceDownDist / bodyHeight;
        if (faceRatio < 0.2) {
            return { valid: false, reason: 'Face not pointing down' };
        }
    }

    return { valid: true, reason: 'Valid push-up position' };
}

// Main Application
class PushupApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.counter = new PushupCounter();
        this.pose = null;
        this.camera = null;
        this.facingMode = 'user'; // 'user' for front camera, 'environment' for back

        this.positionValid = false;
        this.warningFrames = 0;
        this.scorePopupFrames = 0;

        this.initUI();
        this.initPose();
    }

    initUI() {
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.counter.reset();
            this.updateUI();
        });

        document.getElementById('flipBtn').addEventListener('click', () => {
            this.flipCamera();
        });
    }

    async flipCamera() {
        if (this.camera) {
            this.camera.stop();
        }

        this.facingMode = this.facingMode === 'user' ? 'environment' : 'user';
        await this.startCamera();
    }

    async initPose() {
        this.pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });

        this.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.pose.onResults((results) => this.onPoseResults(results));

        await this.startCamera();

        document.getElementById('loadingScreen').style.display = 'none';
    }

    async startCamera() {
        const constraints = {
            video: {
                facingMode: this.facingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = stream;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    resolve();
                };
            });

            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            this.camera = new Camera(this.video, {
                onFrame: async () => {
                    await this.pose.send({ image: this.video });
                },
                width: 1280,
                height: 720
            });

            this.camera.start();
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Could not access camera. Please grant camera permissions.');
        }
    }

    onPoseResults(results) {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!results.poseLandmarks) {
            return;
        }

        const landmarks = results.poseLandmarks;

        // Draw skeleton
        this.drawSkeleton(landmarks);

        // Get keypoints
        const nose = landmarks[0];
        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];
        const leftElbow = landmarks[13];
        const rightElbow = landmarks[14];
        const leftWrist = landmarks[15];
        const rightWrist = landmarks[16];
        const leftHip = landmarks[23];
        const rightHip = landmarks[24];
        const leftKnee = landmarks[25];
        const rightKnee = landmarks[26];
        const leftAnkle = landmarks[27];
        const rightAnkle = landmarks[28];

        // Calculate midpoints
        const shoulder = {
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2
        };
        const hip = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        };
        const knee = {
            x: (leftKnee.x + rightKnee.x) / 2,
            y: (leftKnee.y + rightKnee.y) / 2
        };
        const ankle = {
            x: (leftAnkle.x + rightAnkle.x) / 2,
            y: (leftAnkle.y + rightAnkle.y) / 2
        };

        // Validate position
        const positionCheck = isInPushupPosition(shoulder, hip, knee, ankle, nose);
        this.positionValid = positionCheck.valid;

        if (!positionCheck.valid) {
            this.warningFrames = 90; // 3 seconds at 30fps
            document.getElementById('warningReason').textContent = positionCheck.reason;
        }

        // Calculate angles
        const elbowL = calculateAngle(leftShoulder, leftElbow, leftWrist);
        const elbowR = calculateAngle(rightShoulder, rightElbow, rightWrist);
        const elbowAvg = (elbowL + elbowR) / 2;

        const kneeL = calculateAngle(leftHip, leftKnee, leftAnkle);
        const kneeR = calculateAngle(rightHip, rightKnee, rightAnkle);
        const kneeAvg = (kneeL + kneeR) / 2;

        const armBackDiffL = armTorsoAngleDiff(leftShoulder, leftElbow, leftHip);
        const armBackDiffR = armTorsoAngleDiff(rightShoulder, rightElbow, rightHip);
        const armBackDiff = (armBackDiffL + armBackDiffR) / 2;

        // Update counter (only if in valid position)
        if (this.positionValid) {
            const repCompleted = this.counter.update(elbowAvg, armBackDiff);

            if (repCompleted) {
                this.showScorePopup(85); // Mock score for now
            }
        }

        // Update UI
        this.updateUI(elbowAvg, armBackDiff, kneeAvg);
    }

    drawSkeleton(landmarks) {
        const connections = [
            [11, 13], [13, 15], // Left arm
            [12, 14], [14, 16], // Right arm
            [11, 12], // Shoulders
            [11, 23], [12, 24], // Torso
            [23, 24], // Hips
            [23, 25], [25, 27], // Left leg
            [24, 26], [26, 28]  // Right leg
        ];

        this.ctx.strokeStyle = '#00ff88';
        this.ctx.lineWidth = 4;

        connections.forEach(([startIdx, endIdx]) => {
            const start = landmarks[startIdx];
            const end = landmarks[endIdx];

            if (start.visibility > 0.5 && end.visibility > 0.5) {
                this.ctx.beginPath();
                this.ctx.moveTo(start.x * this.canvas.width, start.y * this.canvas.height);
                this.ctx.lineTo(end.x * this.canvas.width, end.y * this.canvas.height);
                this.ctx.stroke();
            }
        });

        // Draw keypoints
        landmarks.forEach((landmark) => {
            if (landmark.visibility > 0.5) {
                this.ctx.fillStyle = '#00ff88';
                this.ctx.beginPath();
                this.ctx.arc(
                    landmark.x * this.canvas.width,
                    landmark.y * this.canvas.height,
                    6,
                    0,
                    2 * Math.PI
                );
                this.ctx.fill();
            }
        });
    }

    updateUI(elbowAngle = null, armBackAngle = null, kneeAngle = null) {
        // Rep count
        document.getElementById('repCount').textContent = this.counter.reps;

        // Position indicator
        const posIndicator = document.getElementById('positionIndicator');
        if (this.positionValid) {
            posIndicator.innerHTML = '<span class="position-valid">✓ READY</span>';
        } else {
            posIndicator.innerHTML = '<span class="position-invalid">✗ INVALID POS</span>';
        }

        // Metrics
        if (elbowAngle !== null) {
            document.getElementById('elbowAngle').textContent = `${Math.round(elbowAngle)}°`;
        }
        if (armBackAngle !== null) {
            const elem = document.getElementById('armBackAngle');
            const isParallel = armBackAngle <= this.counter.parallelThreshold;
            elem.textContent = `${Math.round(armBackAngle)}°${isParallel ? ' [PARALLEL]' : ''}`;
            elem.className = `metric-value ${isParallel ? 'metric-good' : 'metric-warning'}`;
        }
        if (kneeAngle !== null) {
            const elem = document.getElementById('kneeAngle');
            const isStraight = kneeAngle >= 160;
            elem.textContent = `${Math.round(kneeAngle)}°`;
            elem.className = `metric-value ${isStraight ? 'metric-good' : 'metric-warning'}`;
        }

        // Warning banner
        const warningBanner = document.getElementById('warningBanner');
        if (this.warningFrames > 0) {
            warningBanner.classList.add('show');
            this.warningFrames--;
        } else {
            warningBanner.classList.remove('show');
        }

        // Score popup
        if (this.scorePopupFrames > 0) {
            this.scorePopupFrames--;
            if (this.scorePopupFrames === 0) {
                document.getElementById('scorePopup').classList.remove('show');
            }
        }
    }

    showScorePopup(score) {
        document.getElementById('scoreValue').textContent = score;
        document.getElementById('scorePopup').classList.add('show');
        this.scorePopupFrames = 90; // Show for 3 seconds
    }
}

// Start the app when page loads
window.addEventListener('load', () => {
    new PushupApp();
});