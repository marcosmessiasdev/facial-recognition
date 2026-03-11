# ADR-001: Use ONNX Runtime for Model Inference

## Status
Accepted

## Context
The system requires running multiple deep learning models (Detection, Recognition, Attribute classification) in real-time. We need a performant, cross-platform inference engine that supports GPU acceleration and handles models from different training frameworks (PyTorch, Keras).

## Decision
All machine learning components will utilize the Microsoft ONNX Runtime. Models must be provided or converted to the `.onnx` format.

## Consequences
- **Pros**: High performance, standardized API across different model types, easy integration with C#, support for DirectML/CUDA.
- **Cons**: Requires explicit preprocessing steps (mean subtraction, normalization) implemented manually for each model type.
