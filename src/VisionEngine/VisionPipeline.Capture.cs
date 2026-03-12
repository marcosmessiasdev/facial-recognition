using FramePipeline;

namespace VisionEngine;

public partial class VisionPipeline
{
    /// <summary>
    /// Handles incoming raw video frames from the capture service.
    /// </summary>
    private void OnRawFrameArrived(object? sender, (byte[] data, int width, int height, int stride) args)
    {
        // Producer — drop frame if queue is full to keep latency minimal
        if (_frameQueue.Count < _frameQueue.BoundedCapacity)
        {
            VisionFrame visionFrame = new(args.data, args.width, args.height, args.stride);
            _ = _frameQueue.TryAdd(visionFrame);
        }
    }
}

