# System Data Flow

## Overview
The platform connects separate concurrent inputs—display/video capture and system audio—synchronizing them continuously for real-time analysis.

## Sequence

```
Client/UI Selection (HWND)
     │
     ▼
WindowCapture (Windows.Graphics.Capture via COM/WinRT)
     │   (Generates 30-60 FPS DirectX surfaces)
     ▼
VisionPipeline (Main Processing Loop)
     │   (Subscribes to FrameArrived / AudioArrived)
     ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
     ▼                                                        ▼
Audio processing (NAudio loopback)                     Video (OpenCvSharp Mat extraction)
     │                                                        │
     ▼                                                        ▼
SileroVad (Voice Detection)                            FaceDetection (SCRFD ONNX)
     │ (Filters out silence)                                  │
     ▼                                                        ▼
Audio Buffer (PCM 16k)                                 FaceTracker (SORT / ByteTrack)
     │                                                        │
     ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ (Joins Audio/Video Sync on Time)       
     ▼                                                        ▼
Whisper Transcription (STT)                            TalkNetAsdModel (Active Speaker)
     │                                                        │
     ▼                                                        ▼
MeetingAnalyticsEngine ◄─────────────────────────────── FaceRecognition / FaceAttributes
     │ (Aggregates track & transcript data)                   │
     ▼                                                        ▼
OverlayRenderer (WPF UI drawing bounds and text)─────── IdentityStore (Comparing embeddings)
```

## Validation & Business Logic
*   **Window Bounds Sync**: The `OverlayRenderer` intercepts `GetWindowRect` to continuously align perfectly on top of the moving parent window, avoiding off-by-one pixel jumps.
*   **Audio/Visual Timing**: `VisionPipeline` maps monotonic `DateTime.UtcNow` to hardware audio clock ticks, guaranteeing that lip-sync (Active Speaker Detection via TalkNet/Mouth Ratio) and audio chunks don't drift apart.
*   **Pipeline Throttling**: The visual queue forces `if (queue.Count > max) queue.Dequeue()` when CPU inference is too slow, dropping intermediate frames instead of freezing memory/UI.

## Persistence
*   **JSON Exporter**: The `MeetingAnalyticsEngine` uses `System.Text.Json` to write the final domain object (`MeetingSession`) natively to the `logs/` directory upon stopping the pipeline, acting as the final record of events.
