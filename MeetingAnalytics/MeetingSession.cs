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

    /// <summary>
    /// Visual-only speaking segments derived from the active face track (useful for debugging A/V association).
    /// </summary>
    public required List<SpeakerSegment> VisualSegments { get; init; }

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

    /// <summary>Count of speaking turns (segment starts) by speaker.</summary>
    public required Dictionary<string, int> TurnsBySpeaker { get; init; }

    /// <summary>Simple participation score (transparent heuristic) by speaker.</summary>
    public required Dictionary<string, double> ParticipationScoreBySpeaker { get; init; }

    /// <summary>Conversation graph edges (who follows who in turn-taking).</summary>
    public required List<ConversationEdge> ConversationGraph { get; init; }
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

    /// <summary>The visual track ID if this segment was derived from a face track.</summary>
    public int? TrackId { get; set; }

    /// <summary>The audio speaker/cluster ID if this segment was derived from diarization.</summary>
    public int? AudioSpeakerId { get; set; }
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

    /// <summary>The resolved visual track ID for this utterance, when available.</summary>
    public int? TrackId { get; set; }

    /// <summary>The resolved audio speaker/cluster ID for this utterance, when available.</summary>
    public int? AudioSpeakerId { get; set; }
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

public sealed class ConversationEdge
{
    public required string FromSpeakerKey { get; init; }
    public required string ToSpeakerKey { get; init; }
    public int Count { get; init; }
}
