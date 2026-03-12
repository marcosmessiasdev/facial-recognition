using FacialRecognition.Domain;
using FramePipeline;

namespace VisionEngine.Stages;

internal sealed class FrameContext
{
    public required VisionFrame Frame { get; init; }
    public required long FrameCount { get; init; }

    public List<BoundingBox> Boxes { get; set; } = [];
    public List<Track> Tracks { get; set; } = [];

    public DateTime NowUtc { get; init; }

    // Audio-derived state (owned by VisionPipeline.Audio.cs)
    public bool VadSpeechActive { get; init; }
    public float VadSpeechProb { get; init; }
    public TimeSpan LastAudioOffset { get; init; }
    public int? ActiveAudioSpeakerId { get; init; }
    public float ActiveAudioSpeakerConfidence { get; init; }

    public required Func<TimeSpan, TimeSpan, (bool Ok, float[] Audio)> TryGetAudioWindow { get; init; }

    public int? ActiveSpeakerTrackId { get; set; }
}

