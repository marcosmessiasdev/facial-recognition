# Data Flow Documentation

This document explains the typical lifecycle of data from capture to analytics.

## Visual Pipeline Flow
1. **Window Selection**: User selects a window handle (HWND).
2. **Frame Acquisition**: `GraphicsCaptureService` emits a raw buffer.
3. **Frame Wrapper**: `VisionFrame` converts the raw buffer into an OpenCV `Mat` (BGRA).
4. **Face Detection**: `FaceDetector` identifies face regions and landmarks.
5. **Tracking Update**: `FaceTracker` matches detections to existing `Track` objects.
6. **Throttled Analysis**:
   - If a track is new: `ArcFaceRecognizer` extracts identity.
   - Every N frames: `EmotionClassifier`, `AgeClassifier`, and `GenderClassifier` update track metadata.
7. **Mouth Analysis**: `MouthMotionAnalyzer` calculates delta scores for each track's mouth region.

## Audio Pipeline Flow
1. **Capture**: `LoopbackAudioCapture` records system output using WASAPI.
2. **Normalization**: Audio is resampled to 16kHz Mono.
3. **VAD Inference**: `SileroVad` calculates the probability of human speech in 512-sample segments.
4. **Speech Status**: A boolean "SpeechActive" flag is emitted.

## Merged Analytics Flow
1. **Active Speaker Identification**: `ActiveSpeakerDetector` receives `Track` list + `SpeechActive`. It selects the track with the highest mouth motion among those exceeding thresholds.
2. **Session Persistence**: `MeetingAnalyticsEngine` records the start and end of speaking segments.
3. **Final Report**: When stopped, a JSON summary is generated with aggregate participation statistics.

## Validation & Business Logic
- **Identity Threshold**: Recognition requires a similarity score > 0.40 (configurable).
- **VAD Threshold**: Speech is only considered valid if probability > 0.60.
- **Persistence**: Database schema is ensured on initialization via EF Core `EnsureCreated()`.
