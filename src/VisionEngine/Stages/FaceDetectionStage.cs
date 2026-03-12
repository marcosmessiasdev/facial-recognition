using FaceDetection;
using Logging;

namespace VisionEngine.Stages;

internal sealed class FaceDetectionStage(FaceDetector detector) : IFrameStage
{
    public void Process(FrameContext ctx)
    {
        ctx.Boxes = detector.Detect(ctx.Frame);

        // Log face count every frame for E2E test log validation
        if (ctx.Boxes.Count > 0 || ctx.FrameCount % 30 == 0)
        {
            AppLogger.Instance.Debug("Faces detected: {Count} | Frame: {Frame}", ctx.Boxes.Count, ctx.FrameCount);
        }
    }
}

