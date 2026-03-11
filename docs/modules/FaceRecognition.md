# Module: FaceRecognition

## Purpose
Transforms cropped facial images into unique mathematical signatures (embeddings) for identity verification.

## Key Components
- **ArcFaceRecognizer**: Implements the ArcFace ResNet-100 architecture inference.

## Responsibilities
- Crop and align facial images to the standard 112x112 input size.
- Perform pixel normalization ((pixel - 127.5) / 128.0).
- Extract 512-dimensional feature vectors.
- L2-Normalize output vectors to unit length for similarity calculations.

## Dependencies
- **Microsoft.ML.OnnxRuntime**: Executes the model.
- **OpenCvSharp**: Handles high-performance image resizing and color conversion.

## Dependencies (Architectural)
- Used by the **VisionEngine** to identify participants against the **IdentityStore**.
