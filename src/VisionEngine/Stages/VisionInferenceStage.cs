using AgeAnalysis;
using Config;
using EmotionAnalysis;
using FaceAttributes;
using FaceRecognition;
using GenderAnalysis;
using IdentityStore;
using Logging;
using MeetingAnalytics;
using OpenCvSharp;

namespace VisionEngine.Stages;

internal sealed class VisionInferenceStage(
    AppConfig cfg,
    PersonRepository personRepo,
    ArcFaceRecognizer? recognizer,
    EmotionClassifier? emotion,
    GenderAgeClassifier? genderAge,
    GenderClassifier? gender,
    AgeClassifier? age,
    MeetingAnalyticsEngine? analytics) : IFrameStage
{
    public void Process(FrameContext ctx)
    {
        if (ctx.Frame.Mat == null || ctx.Frame.Mat.Empty())
        {
            return;
        }

        foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
        {
            t.FramesSinceRecognition++;
            t.FramesSinceEmotion++;
            t.FramesSinceEmotionDebugLog++;
            t.FramesSinceGender++;
            t.FramesSinceAge++;
        }

        RunRecognition(ctx);
        RunEmotion(ctx);
        RunAttributes(ctx);
    }

    private void RunRecognition(FrameContext ctx)
    {
        if (recognizer == null || ctx.Tracks.Count == 0)
        {
            return;
        }

        foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
        {
            if (t.FramesSinceRecognition < cfg.RecognitionIntervalFrames)
            {
                continue;
            }

            try
            {
                Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                rect = ClampRect(rect, ctx.Frame.Mat!.Width, ctx.Frame.Mat.Height);
                if (rect.Width < 16 || rect.Height < 16)
                {
                    continue;
                }

                using Mat face = new(ctx.Frame.Mat, rect);
                using Mat crop = face.Clone();
                float[] emb = recognizer.GetEmbedding(crop);

                (Person? person, float sim) = personRepo.FindBestMatch(emb, threshold: (float)cfg.RecognitionThreshold);
                if (person != null && !string.IsNullOrWhiteSpace(person.Name))
                {
                    string newName = $"{person.Name} ({sim:P0})";
                    bool changed = string.IsNullOrWhiteSpace(t.PersonName) ||
                                   !t.PersonName.StartsWith(person.Name, StringComparison.OrdinalIgnoreCase);
                    t.PersonName = newName;
                    if (changed)
                    {
                        AppLogger.Instance.Information("Recognized: {Name} sim={Sim:0.00} track={TrackId}", person.Name, sim, t.Id);
                    }
                }
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Warning(ex, "Recognition inference error");
            }
            finally
            {
                t.FramesSinceRecognition = 0;
            }
        }
    }

    private void RunEmotion(FrameContext ctx)
    {
        if (emotion == null)
        {
            return;
        }

        foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
        {
            if (t.FramesSinceEmotion < cfg.EmotionInterval)
            {
                continue;
            }

            try
            {
                Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                rect = ExpandRect(rect, cfg.EmotionCropMarginRatio, ctx.Frame.Mat!.Width, ctx.Frame.Mat.Height);
                if (rect.Width < 16 || rect.Height < 16)
                {
                    continue;
                }

                using Mat face = new(ctx.Frame.Mat, rect);
                using Mat crop = face.Clone();

                float[] probs = emotion.GetProbabilities(crop);
                (Emotion e1, float p1, Emotion e2, float p2) = Top2Emotion(probs);

                t.EmotionLabel = $"{e1} {p1:P0}  P2 {e2} {p2:P0}";
                analytics?.AddEmotionSample(t, e1.ToString(), p1, ctx.NowUtc);

                if (cfg.EmotionDebugLogProbs &&
                    t.FramesSinceEmotionDebugLog >= Math.Max(1, cfg.EmotionDebugLogEveryNFrames))
                {
                    t.FramesSinceEmotionDebugLog = 0;
                    AppLogger.Instance.Debug("Emotion probs track={TrackId}: {Vec}", t.Id, FormatEmotionProbs(probs));
                }
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Debug(ex, "Emotion inference failed (best-effort)");
            }
            finally
            {
                t.FramesSinceEmotion = 0;
            }
        }
    }

    private void RunAttributes(FrameContext ctx)
    {
        if (genderAge != null)
        {
            foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
            {
                if (t.FramesSinceGender < cfg.AttributesInterval)
                {
                    continue;
                }

                try
                {
                    (string genderLabel, float genderConfidence, string ageLabel)? res = genderAge.Predict(ctx.Frame.Mat!, t.Box);
                    if (res.HasValue)
                    {
                        t.GenderLabel = $"{res.Value.genderLabel} {res.Value.genderConfidence:P0}";
                        t.AgeLabel = $"Age {res.Value.ageLabel}";
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "GenderAge inference failed (best-effort)");
                }
                finally
                {
                    t.FramesSinceGender = 0;
                    t.FramesSinceAge = 0;
                }
            }

            return;
        }

        if (gender != null)
        {
            foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
            {
                if (t.FramesSinceGender < cfg.GenderInterval)
                {
                    continue;
                }

                try
                {
                    Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                    rect = ClampRect(rect, ctx.Frame.Mat!.Width, ctx.Frame.Mat.Height);
                    if (rect.Width < 16 || rect.Height < 16)
                    {
                        continue;
                    }

                    using Mat face = new(ctx.Frame.Mat, rect);
                    using Mat crop = face.Clone();
                    (GenderAppearance g, float conf) = gender.Classify(crop);
                    t.GenderLabel = $"{g} {conf:P0}";
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "Gender inference failed (best-effort)");
                }
                finally
                {
                    t.FramesSinceGender = 0;
                }
            }
        }

        if (age != null)
        {
            foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
            {
                if (t.FramesSinceAge < cfg.AgeInterval)
                {
                    continue;
                }

                try
                {
                    Rect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                    rect = ClampRect(rect, ctx.Frame.Mat!.Width, ctx.Frame.Mat.Height);
                    if (rect.Width < 16 || rect.Height < 16)
                    {
                        continue;
                    }

                    using Mat face = new(ctx.Frame.Mat, rect);
                    using Mat crop = face.Clone();
                    (_, string label, float conf) = age.Classify(crop);
                    t.AgeLabel = $"Age {label} {conf:P0}";
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "Age inference failed (best-effort)");
                }
                finally
                {
                    t.FramesSinceAge = 0;
                }
            }
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

    private static Rect ExpandRect(Rect r, float marginRatio, int w, int h)
    {
        if (marginRatio <= 0f)
        {
            return ClampRect(r, w, h);
        }

        int mx = (int)MathF.Round(r.Width * marginRatio);
        int my = (int)MathF.Round(r.Height * marginRatio);
        Rect expanded = new(r.X - mx, r.Y - my, r.Width + (2 * mx), r.Height + (2 * my));
        return ClampRect(expanded, w, h);
    }

    private static (Emotion e1, float p1, Emotion e2, float p2) Top2Emotion(float[] probs)
    {
        if (probs.Length == 0)
        {
            return (Emotion.Neutral, 0f, Emotion.Neutral, 0f);
        }

        int n = Math.Min(probs.Length, 8);
        int best1 = 0;
        int best2 = 0;
        float pBest1 = probs[0];
        float pBest2 = float.NegativeInfinity;

        for (int i = 1; i < n; i++)
        {
            float p = probs[i];
            if (p > pBest1)
            {
                best2 = best1;
                pBest2 = pBest1;
                best1 = i;
                pBest1 = p;
            }
            else if (p > pBest2)
            {
                best2 = i;
                pBest2 = p;
            }
        }

        return ((Emotion)best1, float.IsFinite(pBest1) ? pBest1 : 0f, (Emotion)best2, float.IsFinite(pBest2) ? pBest2 : 0f);
    }

    private static string FormatEmotionProbs(float[] probs)
    {
        int n = Math.Min(probs.Length, 8);
        if (n <= 0)
        {
            return string.Empty;
        }

        string[] parts = new string[n];
        for (int i = 0; i < n; i++)
        {
            parts[i] = $"{(Emotion)i}={probs[i]:0.000}";
        }

        return string.Join(" ", parts);
    }
}

