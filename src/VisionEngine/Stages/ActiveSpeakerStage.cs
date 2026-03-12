using Config;
using SpeakerDetection;

namespace VisionEngine.Stages;

internal sealed class ActiveSpeakerStage(AppConfig cfg, ActiveSpeakerDetector speaker) : IFrameStage
{
    public void Process(FrameContext ctx)
    {
        ctx.ActiveSpeakerTrackId = speaker.Update(ctx.Tracks, ctx.VadSpeechActive, cfg.EnableVisualSpeakerFallback, ctx.NowUtc);

        foreach (FacialRecognition.Domain.Track t in ctx.Tracks)
        {
            t.IsSpeaking = ctx.ActiveSpeakerTrackId.HasValue && t.Id == ctx.ActiveSpeakerTrackId.Value;
        }
    }
}

