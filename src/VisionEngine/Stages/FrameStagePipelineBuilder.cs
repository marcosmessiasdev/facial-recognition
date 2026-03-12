using Config;
using FaceTracking;
using IdentityStore;
using MeetingAnalytics;
using SpeakerDetection;
using VisionEngine.Services;

namespace VisionEngine.Stages;

[System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1812:Avoid uninstantiated internal classes", Justification = "Instantiated via dependency injection.")]
internal sealed class FrameStagePipelineBuilder(
    AppConfig cfg,
    IVisionModelProvider models,
    FaceTracker tracker,
    PersonRepository personRepo,
    MeetingAnalyticsEngine analytics,
    MouthMotionAnalyzer mouthMotion,
    ActiveSpeakerDetector activeSpeaker) : IFrameStagePipelineBuilder
{
    public IReadOnlyList<IFrameStage> Build()
    {
        List<IFrameStage> stages =
        [
            new FaceDetectionStage(models),
            new FaceTrackingStage(tracker),
        ];

        if (cfg.EnableTalkNetAsd && models.TalkNetAsd != null)
        {
            stages.Add(new TalkNetAsdStage(cfg, models));
        }

        stages.Add(new MouthMotionStage(cfg, mouthMotion, models));
        stages.Add(new VisionInferenceStage(cfg, models, personRepo, analytics));
        stages.Add(new ActiveSpeakerStage(cfg, activeSpeaker));
        stages.Add(new AnalyticsStage(analytics));

        return stages;
    }
}
