# System Architecture

## Overview
The Facial Recognition and Meeting Analytics system is designed to provide real-time, privacy-first (offline) computer vision and audio analysis of meetings or media. It captures the screen and system audio, running multiple deeply-integrated ML models to track speakers, recognize identities, classify emotions, and transcribe speech concurrently.

## Architectural Style
The project follows a **Component-Based Pipeline Architecture** layered over a continuous execution loop. It leans on the **Pipes and Filters** pattern for data flow (audio and video frames) while isolating responsibilities by domain (Detection, Tracking, Audio, Transcription).

## Layers

### 1. Presentation & UI Layer
- **App (`MainWindow`)**: Orchestrates the application lifecycle, taking user input (target window selection).
- **OverlayRenderer**: A high-performance, click-through WPF transparent window that anchors to the target capture window and renders bounding boxes using WPF drawing primitives.

### 2. Application Layer (Orchestration)
- **VisionEngine (`VisionPipeline`)**: The core orchestrator. Manages thread safety, frame queues, component lifecycles, and synchronization of the audio/video streams. Ensures that if the CPU/GPU workload exceeds frame rate, frames are dropped gracefully rather than locking the UI.

### 3. Machine Learning & Inference Layer
Independent domain modules that wrap `Microsoft.ML.OnnxRuntime`.
- **FaceDetection**: Yolo/SCRFD-based detection.
- **FaceTracking**: SORT/ByteTrack methodologies modified for real-time bounding box association.
- **FaceRecognition**: ArcFace embeddings extraction.
- **FaceAttributes & EmotionAnalysis**: Age, gender, and emotion ONNX models.
- **FaceLandmarks**: MediaPipe face mesh extraction for mouth motion analysis.
- **SpeakerDetection**: TalkNet ASD (Active Speaker Detection) merging audio MFCCs with visual frame queues.

### 4. Audio Processing & Transcription Layer
- **AudioProcessing**: NAudio-based WASAPI capture with Silero VAD (Voice Activity Detection).
- **SpeechProcessing**: Whisper.net bindings for STT (Speech-to-Text).

### 5. Domain Analytics Layer
- **MeetingAnalytics**: Centralizes point-in-time detections into a continuous session model. Generates aggregate outputs like speaker dominance, interruptions, and face-to-audio matching.

## Technology Stack
- **Platform**: C#, .NET 8, WPF
- **Inference**: ONNX Runtime (CPU/DirectML/CUDA), Whisper.net
- **Capture API**: Windows.Graphics.Capture, User32, NAudio
- **Video Manipulation**: OpenCvSharp4
