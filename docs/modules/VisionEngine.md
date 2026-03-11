# Module: VisionEngine

## Purpose
The central management layer that integrates all disparate services into a unified vision and audio pipeline.

## Key Components
- **VisionPipeline**: The primary coordinator class.

## Responsibilities
- Initialize all downstream components (`FaceDetector`, `AudioProcessor`, etc.) with configuration settings.
- Manage synchronization between incoming visual frames and audio events.
- Implement throttling logic for heavy tasks (like Recognition and Attribute classification).
- Forward tracking results to the `OverlayRenderer`.
- Manage the global lifecycle (Start/Stop) of the analysis engine.

## Dependencies
- All analytical modules (`FaceDetection`, `Recognition`, `Audio`, etc.).
- **Config**: Receives paths and thresholds.
- **Logging**: Captures operational state changes.
