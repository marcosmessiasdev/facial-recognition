using FaceTracking;

namespace VisionEngine.Stages;

internal sealed class FaceTrackingStage(FaceTracker tracker) : IFrameStage
{
    public void Process(FrameContext ctx)
    {
        ctx.Tracks = tracker.Update(ctx.Boxes);
    }
}

