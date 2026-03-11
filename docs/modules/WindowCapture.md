# Module: WindowCapture

## Purpose
Provides high-performance, low-latency screen capture of specific application windows.

## Key Components
- **GraphicsCaptureService**: Wrapper for the Windows 10/11 Graphics Capture API.
- **Direct3D11Helper**: Utilities for GPU resource allocation and texture management.

## Responsibilities
- Interface with the Windows desktop compositor to acquire window buffers.
- Handle resizing of the target window without crashing the pipeline.
- Extract raw pixels from GPU textures and transfer them to managed memory (system RAM).
- Provide a steady stream of frames to the `VisionEngine`.

## Dependencies
- **Windows.Graphics.Capture**: Modern WinRT capture API.
- **SharpDX / Direct3D11**: For GPU-level texture manipulation.

## Architectural Role
**Infrastructure Layer**: Acts as the primary input sensor for the visual side of the system.
