using Config;
using FaceTracking;
using IdentityStore;
using MeetingAnalytics;
using Microsoft.Extensions.DependencyInjection;
using SpeakerDetection;
using VisionEngine.Services;
using VisionEngine.Stages;

namespace VisionEngine;

public static class VisionEngineServiceCollectionExtensions
{
    public static IServiceCollection AddVisionEngineServices(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        services.AddScoped<IVisionModelProvider, VisionModelProvider>();

        services.AddScoped<FaceTracker>(sp =>
        {
            AppConfig cfg = sp.GetRequiredService<AppConfig>();
            return new FaceTracker(cfg.IouThreshold, cfg.MaxMissedFrames);
        });

        services.AddScoped<PersonRepository>();
        services.AddScoped<MeetingAnalyticsEngine>();
        services.AddScoped<MouthMotionAnalyzer>();
        services.AddScoped<ActiveSpeakerDetector>();

        services.AddScoped<IFrameStagePipelineBuilder, FrameStagePipelineBuilder>();
        services.AddScoped<VisionPipeline>();

        return services;
    }
}

