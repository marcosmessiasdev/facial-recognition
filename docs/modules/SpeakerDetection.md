# Module: SpeakerDetection

## Purpose
Identifies which participant is currently speaking by fusing acoustic and visual data points.

## Key Components
- **MouthMotionAnalyzer**: Calculates pixel-level movement in the mouth region of each track.
- **ActiveSpeakerDetector**: Heuristic engine that selects the most likely speaker.

## Responsibilities
- Monitor mouth regions for each tracked face.
- Smooth motion scores over a temporal window to avoid flickering.
- Fuse active Voice Activity Detection (VAD) status with visual rankings.
- Implement a "persistence" or "hold" logic to maintain speaker status during short pauses.

## Dependencies
- **AudioProcessing**: Source of global VAD status.
- **FaceTracking**: Source of face locations and landmarks.
- **OpenCvSharp**: Used for ROI extraction and frame differencing.

## Constraints
- Visual detection is limited to the camera's field of view; a participant speaking off-camera is attributed to the "None" or "VAD-only" state.
