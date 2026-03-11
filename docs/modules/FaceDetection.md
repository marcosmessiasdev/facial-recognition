# Module: FaceDetection

## Purpose
Responsible for locating human faces and facial landmarks within visual frames.

## Key Components
- **FaceDetector**: The primary entry point that executes ONNX inference.
- **BoundingBox**: Data Transfer Object (DTO) containing coordinates, confidence, and optional landmarks.

## Responsibilities
- Convert OpenCV Mats into formats expected by detection neural networks.
- Execute the SCRFD (Sample and Computation Redistribution for Efficient Face Detection) engine.
- Normalize detection results to absolute pixel coordinates.
- Perform confidence filtering to reduce false positives.

## Dependencies
- **FaceAiSharp**: Underlying wrapper for the SCRFD model.
- **SixLabors.ImageSharp**: Used for memory-safe image manipulation required by the detector.
- **OpenCvSharp**: Source of the raw frame data.
- **Microsoft.ML.OnnxRuntime**: Heavy-lifting inference engine.

## Constraints
- Input images must be reasonably lit; detection accuracy drops significantly in low-light scenarios.
- Designed for detection only; identity verification is handled by the Recognition module.
