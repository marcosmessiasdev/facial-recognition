# System Architecture

## Overview
The Facial Recognition System is a real-time vision and audio analysis application designed to monitor desktop window content, identify participants, and track meeting participation metrics. It follows a modular, decoupled architecture where specialized components (Detection, Recognition, Analysis) are orchestrated by a central Vision Engine.

## Architectural Style
The project uses a **Modular Monolith** approach with clear separation of concerns in a **Layered Architecture**:

### Layers
1. **Presentation Layer (`App`, `OverlayRenderer`)**
   - WPF-based UI for control and transparent overlays for visual feedback.
2. **Logic/Domain Layer (`VisionEngine`, `FaceTracking`, `SpeakerDetection`, `MeetingAnalytics`)**
   - Orchestrates the data flow and implements temporal consistency and session-level logic.
3. **Application/Analysis Layer (`FaceDetection`, `FaceRecognition`, `EmotionAnalysis`, `AgeAnalysis`, `GenderAnalysis`, `FaceAttributes`, `AudioProcessing`)**
   - Specialized services that wrap deep learning models (ONNX) or hardware APIs (WASAPI).
4. **Infrastructure/Data Layer (`Config`, `Logging`, `IdentityStore`, `WindowCapture`, `FramePipeline`)**
   - Provides fundamental services like frame acquisition (`Direct3D11`), persistence (`SQLite/EF Core`), and global configuration.

## Data Flow
1. **Capture**: System audio and desktop window pixels are captured via WASAPI and GraphicsCapture API.
2. **Pre-processing**: Frames are converted to OpenCV `Mat` format and synchronized.
3. **Detection**: Faces are detected using SCRFD; Voice activity is detected using Silero VAD.
4. **Tracking**: Faces are associated across frames to maintain stable IDs.
5. **Analysis**: For each stable track, periodic updates are performed for Identity (ArcFace), Emotion, Age, and Gender.
6. **Analytics**: Speaker detection merges visual mouth motion with VAD to attribute speech to specific participants.
7. **Display**: Results are rendered via a transparent overlay window.

## Technology Stack
- **Runtime**: .NET 8 / .NET 9 (WPF for UI)
- **Computer Vision**: OpenCV (OpenCvSharp4), FaceAiSharp (SCRFD)
- **Machine Learning**: ONNX Runtime (Microsoft.ML.OnnxRuntime)
- **Audio**: NAudio (WASAPI Loopback)
- **Imaging**: SixLabors.ImageSharp
- **Persistence**: Entity Framework Core with SQLite
- **Logging**: Serilog
