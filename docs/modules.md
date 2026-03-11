# System Modules

The system is partitioned into independent modules to ensure high maintainability and testability.

## Vision Modules

### WindowCapture
Responsible for acquiring raw pixel data from active desktop windows using the modern Windows Graphics Capture API and Direct3D11 for GPU-accelerated retrieval.

### FaceDetection
Wraps SCRFD (Sample and Computation Redistribution for Efficient Face Detection) models to locate faces and landmarks in image frames.

### FaceTracking
Implements temporal association logic using Intersection over Union (IoU) to maintain stable identifiers for individuals as they move.

### FaceRecognition
Extracts 512-dimensional feature embeddings using ArcFace models for identity verification.

## Analytical Modules

### EmotionAnalysis
Classifies facial expressions into categories (Happy, Sad, Angry, etc.) using the FER+ dataset patterns.

### Age & Gender Analysis
Separate modules (`AgeAnalysis`, `GenderAnalysis`, `FaceAttributes`) that utilize classification models to estimate biological attributes.

### SpeakerDetection
Combines visual cues from `MouthMotionAnalyzer` with audio status from `AudioProcessing` to identify the current speaker.

## Support Modules

### IdentityStore
Manages the registration and persistence of known individuals using SQLite and EF Core. Supports identity lookup via vector similarity.

### AudioProcessing
Captures system loopback audio and performs Voice Activity Detection (VAD) using Silero.

### MeetingAnalytics
Aggregates session data to calculate participation percentage, speaking time, and interruption counts.

### OverlayRenderer
Renders real-time metadata (bounding boxes, names, status) directly over the target window for user feedback.

### VisionEngine
The central orchestrator that manages the lifecycle of all vision and audio components.
