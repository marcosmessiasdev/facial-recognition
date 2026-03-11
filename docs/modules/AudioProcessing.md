# Module: AudioProcessing

## Purpose
Captures system audio loopback and identifies periods of human speech.

## Key Components
- **LoopbackAudioCapture**: WASAPI-based audio stream listener.
- **SileroVad**: State-of-the-art Voice Activity Detector (VAD) wrapper.
- **AudioFrameEventArgs**: Pass-over object for audio buffers.

## Responsibilities
- Record system-wide audio (DirectX/Windows default endpoint).
- Resample incoming audio to 16,000 Hz / Mono for compatibility with VAD models.
- Apply LSTM-based Silero models to calculate speech probability in 32ms real-time windows.
- Emit events when speech starts or stops.

## Dependencies
- **NAudio**: For WASAPI interop and Resampling.
- **Microsoft.ML.OnnxRuntime**: Executes the Silero LSTM model.
