using Logging;
using MeetingAnalytics;

namespace VisionEngine.Stages;

internal sealed class AnalyticsStage(MeetingAnalyticsEngine analytics) : IFrameStage
{
    private DateTime _lastUpdateUtc = DateTime.MinValue;

    public void Process(FrameContext ctx)
    {
        // Throttle updates a bit to avoid excessive segment churn.
        if ((ctx.NowUtc - _lastUpdateUtc) <= TimeSpan.FromMilliseconds(120))
        {
            return;
        }

        _lastUpdateUtc = ctx.NowUtc;

        try
        {
            analytics.Update(ctx.Tracks, ctx.ActiveSpeakerTrackId, ctx.NowUtc);

            // Record audio↔face co-occurrence for mapping diarization clusters to visual tracks.
            if (ctx.ActiveAudioSpeakerId.HasValue && ctx.ActiveSpeakerTrackId.HasValue)
            {
                analytics.ObserveAudioToFaceCooccurrence(ctx.ActiveAudioSpeakerId.Value, ctx.ActiveSpeakerTrackId.Value);
            }
        }
        catch (Exception ex)
        {
            AppLogger.Instance.Debug(ex, "Analytics update failed (best-effort)");
        }
    }
}

