namespace MeetingAnalytics;

/// <summary>
/// Data container for a complete meeting session and its computed statistics.
/// </summary>
public sealed class MeetingSession
{
    /// <summary>The UTC timestamp when the session started.</summary>
    public required DateTime StartedAtUtc { get; init; }

    /// <summary>The UTC timestamp when the session ended.</summary>
    public DateTime? EndedAtUtc { get; set; }

    /// <summary>The chronological list of speaking segments detected during the session.</summary>
    public required List<SpeakerSegment> Segments { get; init; }

    /// <summary>Aggregated total speaking time (in seconds) mapped by speaker key.</summary>
    public required Dictionary<string, double> SpeakingTimeSecondsBySpeaker { get; init; }

    /// <summary>Count of interruptions mapped by the person who interrupted.</summary>
    public required Dictionary<string, int> InterruptionsBySpeaker { get; init; }

    /// <summary>Transcribed utterances aligned to the best matching speaker segment.</summary>
    public required List<Utterance> Utterances { get; init; }

    /// <summary>Audio-only diarization segments.</summary>
    public required List<AudioSpeakerSegment> AudioSpeakerSegments { get; init; }

    /// <summary>Best-effort mapping from audio speaker IDs to face tracks/identities.</summary>
    public required Dictionary<int, SpeakerFaceMapping> AudioSpeakerToFace { get; init; }
}

/// <summary>
/// Represents a contiguous block of time where a specific person was identified as speaking.
/// </summary>
public sealed class SpeakerSegment
{
    /// <summary>The start of the speaking segment.</summary>
    public required DateTime StartUtc { get; init; }

    /// <summary>The end of the speaking segment, or null if currently active.</summary>
    public DateTime? EndUtc { get; set; }

    /// <summary>The unique key identifying the speaker (e.g., identity or track ID).</summary>
    public required string SpeakerKey { get; init; } // e.g. "Track:12" or "Name:Maria"

    /// <summary>The human-readable display name of the speaker, if known.</summary>
    public string? DisplayName { get; set; }
}

/// <summary>
/// Represents a transcribed utterance and the speaker it was assigned to.
/// </summary>
public sealed class Utterance
{
    public required DateTime StartUtc { get; init; }
    public required DateTime EndUtc { get; init; }
    public required string SpeakerKey { get; init; }
    public string? DisplayName { get; set; }
    public required string Text { get; init; }
}

public sealed class AudioSpeakerSegment
{
    public required DateTime StartUtc { get; init; }
    public required DateTime EndUtc { get; init; }
    public required int SpeakerId { get; init; }
    public float Confidence { get; init; }
}

public sealed class SpeakerFaceMapping
{
    public int? TrackId { get; init; }
    public string? DisplayName { get; init; }
    public float Score { get; init; }
}
