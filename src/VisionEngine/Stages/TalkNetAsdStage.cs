using Config;
using FacialRecognition.Domain;
using Logging;
using OpenCvSharp;
using VisionEngine.Services;
using SpeakerDetection;

namespace VisionEngine.Stages;

internal sealed class TalkNetAsdStage(AppConfig cfg, IVisionModelProvider models) : IFrameStage, IDisposable
{
    private readonly Dictionary<int, Queue<Mat>> _framesByTrack = new();
    private DateTime _lastLogUtc = DateTime.MinValue;

    public void Process(FrameContext ctx)
    {
        TalkNetAsdModel? talkNet = models.TalkNetAsd;
        if (talkNet == null || !cfg.EnableTalkNetAsd || ctx.Frame.Mat == null || ctx.Frame.Mat.Empty())
        {
            foreach (Track t in ctx.Tracks)
            {
                t.TalkNetSpeakingProb = 0f;
                t.FramesSinceAsd++;
            }

            Prune(ctx.Tracks);
            return;
        }

        int windowFrames = Math.Clamp(cfg.TalkNetWindowFrames, 5, 60);
        float fps = Math.Clamp(cfg.CaptureFps, 10, 60);
        TimeSpan window = TimeSpan.FromSeconds(windowFrames / fps);

        foreach (Track t in ctx.Tracks)
        {
            t.FramesSinceAsd++;

            if (t.Box.Width <= 2 || t.Box.Height <= 2)
            {
                continue;
            }

            Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
            rect = ClampRect(rect, ctx.Frame.Mat.Width, ctx.Frame.Mat.Height);
            if (rect.Width < 8 || rect.Height < 8)
            {
                continue;
            }

            using Mat face = new(ctx.Frame.Mat, rect);
            using Mat gray = new();
            if (face.Channels() == 4)
            {
                Cv2.CvtColor(face, gray, ColorConversionCodes.BGRA2GRAY);
            }
            else if (face.Channels() == 3)
            {
                Cv2.CvtColor(face, gray, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                face.CopyTo(gray);
            }

            Mat resized = new();
            Cv2.Resize(gray, resized, new OpenCvSharp.Size(112, 112));

            if (!_framesByTrack.TryGetValue(t.Id, out Queue<Mat>? q))
            {
                q = new Queue<Mat>(windowFrames + 2);
                _framesByTrack[t.Id] = q;
            }

            q.Enqueue(resized);
            while (q.Count > windowFrames)
            {
                q.Dequeue().Dispose();
            }
        }

        Prune(ctx.Tracks);

        // Run inference on tracks that have a full window and a recent audio window.
        (bool ok, float[] audio) = ctx.TryGetAudioWindow(ctx.LastAudioOffset, window);
        if (!ok)
        {
            foreach (Track t in ctx.Tracks)
            {
                t.TalkNetSpeakingProb = 0f;
            }
            return;
        }

        foreach (Track t in ctx.Tracks)
        {
            if (!_framesByTrack.TryGetValue(t.Id, out Queue<Mat>? q) || q.Count < windowFrames)
            {
                t.TalkNetSpeakingProb = 0f;
                continue;
            }

            // Throttle per track for performance.
            if (t.FramesSinceAsd < 2)
            {
                continue;
            }

            float[] probs;
            try
            {
                probs = talkNet.Predict(audio, q.ToArray(), fps);
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Debug(ex, "TalkNet ASD inference failed (best-effort)");
                t.TalkNetSpeakingProb = 0f;
                continue;
            }

            float v = probs.Length > 0 ? probs[^1] : 0f;
            float p = (v < 0f || v > 1f) ? Sigmoid(v) : v;
            t.TalkNetSpeakingProb = Math.Clamp(p, 0f, 1f);
            t.FramesSinceAsd = 0;
        }

        if ((ctx.NowUtc - _lastLogUtc) > TimeSpan.FromSeconds(5))
        {
            _lastLogUtc = ctx.NowUtc;
            float max = ctx.Tracks.Count > 0 ? ctx.Tracks.Max(t => t.TalkNetSpeakingProb) : 0f;
            if (max > 0.0001f)
            {
                AppLogger.Instance.Debug("TalkNet ASD maxProb={Prob:0.000}", max);
            }
        }
    }

    private void Prune(IEnumerable<Track> tracks)
    {
        HashSet<int> keep = [.. tracks.Select(t => t.Id)];
        int[] remove = [.. _framesByTrack.Keys.Where(id => !keep.Contains(id))];
        foreach (int id in remove)
        {
            if (_framesByTrack.TryGetValue(id, out Queue<Mat>? q))
            {
                while (q.Count > 0)
                {
                    q.Dequeue().Dispose();
                }
            }

            _ = _framesByTrack.Remove(id);
        }
    }

    private static Rect ClampRect(Rect r, int w, int h)
    {
        int x = Math.Clamp(r.X, 0, Math.Max(0, w - 1));
        int y = Math.Clamp(r.Y, 0, Math.Max(0, h - 1));
        int rw = Math.Clamp(r.Width, 1, w - x);
        int rh = Math.Clamp(r.Height, 1, h - y);
        return new Rect(x, y, rw, rh);
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0)
        {
            float z = MathF.Exp(-x);
            return 1f / (1f + z);
        }
        else
        {
            float z = MathF.Exp(x);
            return z / (1f + z);
        }
    }

    public void Dispose()
    {
        foreach (Queue<Mat> q in _framesByTrack.Values)
        {
            while (q.Count > 0)
            {
                q.Dequeue().Dispose();
            }
        }

        _framesByTrack.Clear();
        GC.SuppressFinalize(this);
    }
}
