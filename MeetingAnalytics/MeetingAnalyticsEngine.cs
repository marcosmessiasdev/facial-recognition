using System.Text.Json;
using FaceTracking;

namespace MeetingAnalytics;

/// <summary>
/// Core engine for tracking speaking segments and calculating meeting analytics.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Provides quantitative insights into meeting participation by tracking who is speaking and for how long.
///
/// Responsibilities:
/// - Monitor transitions between active speakers.
/// - Record historical segments of speaking activity.
/// - Aggregate total speaking time and interruption counts per individual.
/// - Serialize meeting results to JSON for persistent storage and reporting.
///
/// Dependencies:
/// - FaceTracking (Track metadata)
///
/// Architectural Role:
/// Infrastructure Component / Analytics Service.
///
/// Constraints:
/// - Accuracy depends on the stability of face tracking and the precision of the ActiveSpeakerDetector.
/// </remarks>
public sealed class MeetingAnalyticsEngine
{
    private readonly DateTime _startedAtUtc = DateTime.UtcNow;
    private readonly List<SpeakerSegment> _segments = new();
    private readonly List<Utterance> _utterances = new();
    private readonly List<AudioSpeakerSegment> _audioSegments = new();
    private readonly Dictionary<(int SpeakerId, int TrackId), int> _cooccurrence = new();
    private int? _activeTrackId;

    /// <summary>
    /// Updates the engine state based on the current active speaker.
    /// </summary>
    /// <param name="tracks">The list of all faces currently being tracked.</param>
    /// <param name="activeSpeakerTrackId">The unique ID of the track identified as speaking, or null.</param>
    /// <param name="nowUtc">The current UTC timestamp.</param>
    public void Update(IReadOnlyList<Track> tracks, int? activeSpeakerTrackId, DateTime nowUtc)
    {
        if (_activeTrackId == activeSpeakerTrackId)
        {
            return;
        }

        // Close previous segment
        if (_activeTrackId.HasValue)
        {
            SpeakerSegment? last = _segments.LastOrDefault();
            if (last != null && last.EndUtc == null)
            {
                last.EndUtc = nowUtc;
            }
        }

        _activeTrackId = activeSpeakerTrackId;

        if (!_activeTrackId.HasValue)
        {
            return;
        }

        Track? track = tracks.FirstOrDefault(t => t.Id == _activeTrackId.Value);
        string? display = NormalizeDisplayName(track?.PersonName);
        string key = display != null ? $"Name:{display}" : $"Track:{_activeTrackId.Value}";

        _segments.Add(new SpeakerSegment
        {
            StartUtc = nowUtc,
            EndUtc = null,
            SpeakerKey = key,
            DisplayName = display
        });
    }

    /// <summary>
    /// Finalizes the current meeting session and builds an aggregate analytics report.
    /// </summary>
    /// <param name="endedAtUtc">The timestamp when the session ended.</param>
    /// <returns>A MeetingSession object containing summary statistics.</returns>
    public MeetingSession StopAndBuild(DateTime endedAtUtc)
    {
        // Close open segment
        SpeakerSegment? last = _segments.LastOrDefault();
        if (last != null && last.EndUtc == null)
        {
            last.EndUtc = endedAtUtc;
        }

        Dictionary<string, double> speaking = new(StringComparer.OrdinalIgnoreCase);
        foreach (SpeakerSegment seg in _segments)
        {
            if (seg.EndUtc == null)
            {
                continue;
            }

            double dur = (seg.EndUtc.Value - seg.StartUtc).TotalSeconds;
            if (dur <= 0)
            {
                continue;
            }

            _ = speaking.TryGetValue(seg.SpeakerKey, out double cur);
            speaking[seg.SpeakerKey] = cur + dur;
        }

        // Interruptions (simple heuristic): if a speaker starts within 0.5s of previous segment end
        Dictionary<string, int> interruptions = new(StringComparer.OrdinalIgnoreCase);
        for (int i = 1; i < _segments.Count; i++)
        {
            SpeakerSegment prev = _segments[i - 1];
            SpeakerSegment cur = _segments[i];
            if (prev.EndUtc == null)
            {
                continue;
            }

            TimeSpan gap = cur.StartUtc - prev.EndUtc.Value;
            if (gap < TimeSpan.FromMilliseconds(500))
            {
                _ = interruptions.TryGetValue(cur.SpeakerKey, out int count);
                interruptions[cur.SpeakerKey] = count + 1;
            }
        }

        return new MeetingSession
        {
            StartedAtUtc = _startedAtUtc,
            EndedAtUtc = endedAtUtc,
            Segments = [.. _segments],
            SpeakingTimeSecondsBySpeaker = speaking,
            InterruptionsBySpeaker = interruptions,
            Utterances = [.. _utterances],
            AudioSpeakerSegments = [.. _audioSegments],
            AudioSpeakerToFace = BuildAudioSpeakerToFaceMapping()
        };
    }

    /// <summary>
    /// Adds a transcribed utterance to the session and assigns it to the best overlapping speaker segment.
    /// </summary>
    /// <param name="startUtc">Utterance Start timeline UTC.</param>
    /// <param name="endUtc">Utterance End timeline UTC.</param>
    /// <param name="text">The transcribed text string.</param>
    public void AddUtterance(DateTime startUtc, DateTime endUtc, string text)
    {
        if (endUtc <= startUtc)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        SpeakerSegment? best = FindBestOverlappingSpeaker(startUtc, endUtc);
        _utterances.Add(new Utterance
        {
            StartUtc = startUtc,
            EndUtc = endUtc,
            SpeakerKey = best?.SpeakerKey ?? "Unknown",
            DisplayName = best?.DisplayName,
            Text = text.Trim()
        });
    }

    /// <summary>
    /// Aggregates and returns the top speakers by total duration.
    /// </summary>
    /// <param name="nowUtc">The cutoff boundary to measure active open segments.</param>
    /// <param name="top">Maximum length of the resulting list.</param>
    /// <returns>A collection of tuples containing speaker key, display name, and duration in seconds.</returns>
    public IReadOnlyList<(string SpeakerKey, string? DisplayName, double Seconds)> GetSpeakingTimeSoFar(DateTime nowUtc, int top = 3)
    {
        Dictionary<string, double> totals = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, string?> names = new(StringComparer.OrdinalIgnoreCase);

        foreach (SpeakerSegment seg in _segments)
        {
            DateTime end = seg.EndUtc ?? nowUtc;
            double dur = (end - seg.StartUtc).TotalSeconds;
            if (dur <= 0)
            {
                continue;
            }

            _ = totals.TryGetValue(seg.SpeakerKey, out double cur);
            totals[seg.SpeakerKey] = cur + dur;
            if (!names.ContainsKey(seg.SpeakerKey))
            {
                names[seg.SpeakerKey] = seg.DisplayName;
            }
        }

        return [.. totals
            .OrderByDescending(kvp => kvp.Value)
            .Take(Math.Clamp(top, 1, 10))
            .Select(kvp => (kvp.Key, names.TryGetValue(kvp.Key, out string? n) ? n : null, kvp.Value))];
    }

    /// <summary>
    /// Finds the visual speaker segment that has the longest time intersection with the given interval.
    /// </summary>
    /// <param name="startUtc">Start bound of the query window.</param>
    /// <param name="endUtc">End bound of the query window.</param>
    /// <returns>The best matching SpeakerSegment or null if no overlap exists.</returns>
    private SpeakerSegment? FindBestOverlappingSpeaker(DateTime startUtc, DateTime endUtc)
    {
        SpeakerSegment? best = null;
        double bestOverlap = 0;

        foreach (SpeakerSegment seg in _segments)
        {
            DateTime segEnd = seg.EndUtc ?? DateTime.UtcNow;
            if (segEnd <= startUtc || seg.StartUtc >= endUtc)
            {
                continue;
            }

            DateTime overlapStart = seg.StartUtc > startUtc ? seg.StartUtc : startUtc;
            DateTime overlapEnd = segEnd < endUtc ? segEnd : endUtc;
            double overlap = (overlapEnd - overlapStart).TotalSeconds;
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                best = seg;
            }
        }

        return best;
    }

    /// <summary>
    /// Registers a distinct audio speaker block detected by Diarization engines.
    /// </summary>
    /// <param name="startUtc">When the speaker began.</param>
    /// <param name="endUtc">When the speaker stopped.</param>
    /// <param name="speakerId">The ID assigned by the audio clustering algorithm.</param>
    /// <param name="confidence">Score representing model certainty.</param>
    public void AddAudioSpeakerSegment(DateTime startUtc, DateTime endUtc, int speakerId, float confidence)
    {
        if (endUtc <= startUtc)
        {
            return;
        }

        _audioSegments.Add(new AudioSpeakerSegment
        {
            StartUtc = startUtc,
            EndUtc = endUtc,
            SpeakerId = speakerId,
            Confidence = confidence
        });
    }

    /// <summary>
    /// Records a moment where both the visual speaker (trackId) and audio speaker (speakerId) are active concurrently.
    /// </summary>
    /// <param name="speakerId">The Diarization cluster ID.</param>
    /// <param name="trackId">The Vision Face Tracking ID.</param>
    public void ObserveAudioToFaceCooccurrence(int speakerId, int trackId)
    {
        if (speakerId <= 0 || trackId <= 0)
        {
            return;
        }

        (int speakerId, int trackId) key = (speakerId, trackId);
        _ = _cooccurrence.TryGetValue(key, out int c);
        _cooccurrence[key] = c + 1;
    }

    /// <summary>
    /// Estimates which visual track corresponds to which audio cluster ID based on recorded co-occurrence history.
    /// </summary>
    /// <returns>A mapping from Audio Speaker ID to Face Track metadata.</returns>
    private Dictionary<int, SpeakerFaceMapping> BuildAudioSpeakerToFaceMapping()
    {
        // Greedy assignment: for each speakerId pick the trackId with max cooccurrence.
        Dictionary<int, List<KeyValuePair<(int SpeakerId, int TrackId), int>>> bySpeaker = _cooccurrence
            .GroupBy(kvp => kvp.Key.SpeakerId)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(x => x.Value).ToList());

        HashSet<int> usedTracks = new();
        Dictionary<int, SpeakerFaceMapping> map = new();

        foreach (int speakerId in bySpeaker.Keys.OrderBy(k => k))
        {
            List<KeyValuePair<(int SpeakerId, int TrackId), int>> candidates = bySpeaker[speakerId];
            KeyValuePair<(int SpeakerId, int TrackId), int> best = candidates.FirstOrDefault(c => !usedTracks.Contains(c.Key.TrackId));
            if (best.Key.TrackId > 0)
            {
                _ = usedTracks.Add(best.Key.TrackId);
                map[speakerId] = new SpeakerFaceMapping
                {
                    TrackId = best.Key.TrackId,
                    DisplayName = null,
                    Score = best.Value
                };
            }
            else
            {
                map[speakerId] = new SpeakerFaceMapping { TrackId = null, DisplayName = null, Score = 0 };
            }
        }

        return map;
    }

    /// <summary>
    /// Removes similarity scores or appended metadata from a matched person's name string.
    /// </summary>
    /// <param name="personName">The raw formatted name.</param>
    /// <returns>A clean string with the person's name.</returns>
    private static string? NormalizeDisplayName(string? personName)
    {
        if (string.IsNullOrWhiteSpace(personName))
        {
            return null;
        }

        // PersonName often includes similarity, e.g. "Maria (82%)" -> keep just "Maria".
        int idx = personName.IndexOf(" (", StringComparison.Ordinal);
        if (idx > 0)
        {
            personName = personName[..idx];
        }

        return personName.Trim();
    }

    private static readonly JsonSerializerOptions s_jsonOptions = new()
    {
        WriteIndented = true
    };

    /// <summary>
    /// Persists a meeting session to a JSON file on disk.
    /// </summary>
    /// <param name="session">The session data to save.</param>
    /// <param name="baseDir">The root directory for logs.</param>
    /// <returns>The absolute path to the saved file.</returns>
    public static string Persist(MeetingSession session, string baseDir)
    {
        ArgumentNullException.ThrowIfNull(session);

        _ = Directory.CreateDirectory(Path.Combine(baseDir, "logs"));
        string path = Path.Combine(baseDir, "logs", $"meeting_session_{session.StartedAtUtc:yyyyMMdd_HHmmss}.json");

        string json = JsonSerializer.Serialize(session, s_jsonOptions);

        File.WriteAllText(path, json);
        return path;
    }
}
