using Config;
using Logging;
using OpenCvSharp;
using SpeakerDetection;
using VisionEngine.Services;

namespace VisionEngine.Stages;

internal sealed class MouthMotionStage(AppConfig cfg, MouthMotionAnalyzer mouth, IVisionModelProvider models) : IFrameStage
{
    public void Process(FrameContext ctx)
    {
        if (ctx.Frame.Mat == null || ctx.Frame.Mat.Empty())
        {
            foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
            {
                t.MouthMotionScore = 0f;
                t.MouthOpenRatio = 0f;
                t.FramesSinceLandmarks++;
            }

            return;
        }

        var faceMesh = models.FaceMesh;
        bool useFaceMesh = cfg.EnableFaceMeshLandmarks && faceMesh != null;

        foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
        {
            t.FramesSinceLandmarks++;

            Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
            rect = ClampRect(rect, ctx.Frame.Mat.Width, ctx.Frame.Mat.Height);
            if (rect.Width < 16 || rect.Height < 16)
            {
                t.MouthMotionScore = 0f;
                continue;
            }

            using Mat face = new(ctx.Frame.Mat, rect);
            using Mat faceClone = face.Clone();

            Rect? mouthRoi = null;
            float? openRatio = null;

            if (useFaceMesh && t.FramesSinceLandmarks >= cfg.FaceMeshIntervalFrames)
            {
                try
                {
                    if (faceMesh!.TryGetMouthMetrics(faceClone, out float r, out Rect roi))
                    {
                        mouthRoi = roi;
                        openRatio = r;
                        t.MouthOpenRatio = r;
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "FaceMesh mouth metrics failed (best-effort)");
                }

                t.FramesSinceLandmarks = 0;
            }

            float score;
            try
            {
                score = mouth.Update(t.Id, faceClone, mouthRoi, openRatio, ctx.NowUtc);
            }
            catch
            {
                score = 0f;
            }

            t.MouthMotionScore = score;
        }

        mouth.PruneToActiveTracks(ctx.Tracks.Select(t => t.Id));
    }

    private static Rect ClampRect(Rect r, int w, int h)
    {
        int x = Math.Clamp(r.X, 0, Math.Max(0, w - 1));
        int y = Math.Clamp(r.Y, 0, Math.Max(0, h - 1));
        int rw = Math.Clamp(r.Width, 1, w - x);
        int rh = Math.Clamp(r.Height, 1, h - y);
        return new Rect(x, y, rw, rh);
    }
}
