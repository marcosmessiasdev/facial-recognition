Title: ADR-001 - ONNX Runtime for Inter-Process Inference Independence
Status: Accepted

Context:
The platform required executing five+ unique deep learning models (Age, Emotion, Identity, Screen Capture face mapping, Audio VAD). Running native Python bridges or HTTP REST servers incurred severe IPC (Inter-Process Communication) and JSON serialization penalties, dropping the frame rate heavily.

Decision:
The system hosts the models directly in the .NET 8 memory space using `Microsoft.ML.OnnxRuntime`. C# memory arrays (like `float[]` or OpenCV's `Mat`) are wrapped straight into `DenseTensor<float>` objects avoiding cross-boundary copies.

Consequences:
*   Positives: Zero-copy integration, extreme low-latency tracking, simple CPU/GPU pluggable architecture (CUDA/DirectML execution providers).
*   Negatives: Requires exporting PyTorch/Keras models to `.onnx` exactly matching tensor inputs. Lack of advanced standard Python ML libraries for post-processing.
