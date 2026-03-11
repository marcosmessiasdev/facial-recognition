using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    private int? _activeTrackId;

    /// <summary>
    /// Updates the engine state based on the current active speaker.
    /// </summary>
    /// <param name="tracks">The list of all faces currently being tracked.</param>
    /// <param name="activeSpeakerTrackId">The unique ID of the track identified as speaking, or null.</param>
    /// <param name="nowUtc">The current UTC timestamp.</param>
    public void Update(IReadOnlyList<Track> tracks, int? activeSpeakerTrackId, DateTime nowUtc)
    {
        if (_activeTrackId == activeSpeakerTrackId) return;

        // Close previous segment
        if (_activeTrackId.HasValue)
        {
            var last = _segments.LastOrDefault();
            if (last != null && last.EndUtc == null)
                last.EndUtc = nowUtc;
        }

        _activeTrackId = activeSpeakerTrackId;

        if (!_activeTrackId.HasValue)
            return;

        var track = tracks.FirstOrDefault(t => t.Id == _activeTrackId.Value);
        var display = NormalizeDisplayName(track?.PersonName);
        var key = display != null ? $"Name:{display}" : $"Track:{_activeTrackId.Value}";

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
        var last = _segments.LastOrDefault();
        if (last != null && last.EndUtc == null)
            last.EndUtc = endedAtUtc;

        var speaking = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var seg in _segments)
        {
            if (seg.EndUtc == null) continue;
            var dur = (seg.EndUtc.Value - seg.StartUtc).TotalSeconds;
            if (dur <= 0) continue;

            speaking.TryGetValue(seg.SpeakerKey, out var cur);
            speaking[seg.SpeakerKey] = cur + dur;
        }

        // Interruptions (simple heuristic): if a speaker starts within 0.5s of previous segment end
        var interruptions = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 1; i < _segments.Count; i++)
        {
            var prev = _segments[i - 1];
            var cur = _segments[i];
            if (prev.EndUtc == null) continue;

            var gap = cur.StartUtc - prev.EndUtc.Value;
            if (gap < TimeSpan.FromMilliseconds(500))
            {
                interruptions.TryGetValue(cur.SpeakerKey, out var count);
                interruptions[cur.SpeakerKey] = count + 1;
            }
        }

        return new MeetingSession
        {
            StartedAtUtc = _startedAtUtc,
            EndedAtUtc = endedAtUtc,
            Segments = new List<SpeakerSegment>(_segments),
            SpeakingTimeSecondsBySpeaker = speaking,
            InterruptionsBySpeaker = interruptions,
            Utterances = new List<Utterance>(_utterances)
        };
    }

    /// <summary>
    /// Adds a transcribed utterance to the session and assigns it to the best overlapping speaker segment.
    /// </summary>
    public void AddUtterance(DateTime startUtc, DateTime endUtc, string text)
    {
        if (endUtc <= startUtc) return;
        if (string.IsNullOrWhiteSpace(text)) return;

        var best = FindBestOverlappingSpeaker(startUtc, endUtc);
        _utterances.Add(new Utterance
        {
            StartUtc = startUtc,
            EndUtc = endUtc,
            SpeakerKey = best?.SpeakerKey ?? "Unknown",
            DisplayName = best?.DisplayName,
            Text = text.Trim()
        });
    }

    public IReadOnlyList<(string SpeakerKey, string? DisplayName, double Seconds)> GetSpeakingTimeSoFar(DateTime nowUtc, int top = 3)
    {
        var totals = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        var names = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        foreach (var seg in _segments)
        {
            var end = seg.EndUtc ?? nowUtc;
            var dur = (end - seg.StartUtc).TotalSeconds;
            if (dur <= 0) continue;

            totals.TryGetValue(seg.SpeakerKey, out var cur);
            totals[seg.SpeakerKey] = cur + dur;
            if (!names.ContainsKey(seg.SpeakerKey))
                names[seg.SpeakerKey] = seg.DisplayName;
        }

        return totals
            .OrderByDescending(kvp => kvp.Value)
            .Take(Math.Clamp(top, 1, 10))
            .Select(kvp => (kvp.Key, names.TryGetValue(kvp.Key, out var n) ? n : null, kvp.Value))
            .ToArray();
    }

    private SpeakerSegment? FindBestOverlappingSpeaker(DateTime startUtc, DateTime endUtc)
    {
        SpeakerSegment? best = null;
        double bestOverlap = 0;

        foreach (var seg in _segments)
        {
            var segEnd = seg.EndUtc ?? DateTime.UtcNow;
            if (segEnd <= startUtc || seg.StartUtc >= endUtc) continue;

            var overlapStart = seg.StartUtc > startUtc ? seg.StartUtc : startUtc;
            var overlapEnd = segEnd < endUtc ? segEnd : endUtc;
            var overlap = (overlapEnd - overlapStart).TotalSeconds;
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                best = seg;
            }
        }

        return best;
    }

    private static string? NormalizeDisplayName(string? personName)
    {
        if (string.IsNullOrWhiteSpace(personName)) return null;

        // PersonName often includes similarity, e.g. "Maria (82%)" -> keep just "Maria".
        var idx = personName.IndexOf(" (", StringComparison.Ordinal);
        if (idx > 0) personName = personName.Substring(0, idx);
        return personName.Trim();
    }

    /// <summary>
    /// Persists a meeting session to a JSON file on disk.
    /// </summary>
    /// <param name="session">The session data to save.</param>
    /// <param name="baseDir">The root directory for logs.</param>
    /// <returns>The absolute path to the saved file.</returns>
    public string Persist(MeetingSession session, string baseDir)
    {
        Directory.CreateDirectory(Path.Combine(baseDir, "logs"));
        var path = Path.Combine(baseDir, "logs", $"meeting_session_{session.StartedAtUtc:yyyyMMdd_HHmmss}.json");

        var json = JsonSerializer.Serialize(session, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        File.WriteAllText(path, json);
        return path;
    }
}
