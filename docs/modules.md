# System Modules

## App
### FacialRecognitionApp
The entry point of the application. It acts as the Application Controller for user interaction to select the captured window.

## Audio Processing
### LoopbackAudioCapture
Captures system audio via the Windows Audio Session API (WASAPI) using NAudio. Focuses on low-latency, real-time PCM extraction.
### SileroVad
Voice Activity Detection. Detects if human speech is present in small audio chunks to gate resource-intensive ML execution (e.g., stopping transcription during silence).

## Computer Vision
### VisionPipeline
The central orchestrator. Maintains the main asynchronous processing loop, aligning video frames with audio cues, calling independent detection models sequentially.

### FramePipeline
Defines the core `VisionFrame` unit, encapsulating `Mat` objects (from OpenCV) alongside synchronization timestamps to map a frame perfectly back to the system clock.

### FaceDetection
Wraps SCRFD/Yolo models to locate human faces rapidly from the incoming frame buffer.

### FaceTracking
Implements a custom SORT-style tracker to temporally associate detected faces across frames. Vital for assigning an ID to a moving speaker without continually rerunning ArcFace embedding.

### FaceRecognition
Uses ArcFace to generate facial embeddings (512-dimensional vectors) for identity matching over cosine similarity logic.

### FaceAttributes & EmotionAnalysis
Applies secondary classification models (Age, Gender, Emotion) to detected face crops.

### FaceLandmarks
Provides sparse/dense point regression on the face (specifically used for drawing the 3D mesh contour and extracting the inner mouth bounding box).

## Speaker Analysis
### SpeakerDetection
Uses TalkNet (Visual-Audio ASD) and an inner mouth-motion tracking heuristic to calculate the probability of a face actively speaking.
### SpeakerDiarization
Clusters isolated audio embeddings to separate distinct voices even when faces are off-screen.

## Meeting Analytics
### MeetingAnalyticsEngine
Computes meeting-level metrics such as dominant speakers, speaking time distribution, interruption counters, and session timeline events. Emits JSON artifacts upon session stop.

## Transcription
### SpeechProcessing
Wraps Whisper.net. Streams audio into the transcriber (C++ GGML backend) to generate partial and final subtitles synchronously.

## Identity & Config
### IdentityStore
Persistence component for user profiles. Handles comparing incoming ArcFace embeddings with stored templates to consistently label the meeting metadata.
### Config
Initializes configurations via `appsettings.json`, providing dependency paths to the heavy `.onnx` model files.
