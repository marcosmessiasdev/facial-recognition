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
    private readonly List<EmotionSample> _emotionSamples = new();
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
            DisplayName = display,
            TrackId = _activeTrackId.Value,
            AudioSpeakerId = null
        });
    }

    public void AddEmotionSample(Track track, string emotion, float confidence, DateTime whenUtc)
    {
        ArgumentNullException.ThrowIfNull(track);
        if (string.IsNullOrWhiteSpace(emotion))
        {
            return;
        }

        string? display = NormalizeDisplayName(track.PersonName);
        string key = display != null ? $"Name:{display}" : $"Track:{track.Id}";

        _emotionSamples.Add(new EmotionSample
        {
            WhenUtc = whenUtc,
            SpeakerKey = key,
            DisplayName = display,
            TrackId = track.Id,
            Emotion = emotion.Trim(),
            Confidence = Math.Clamp(confidence, 0f, 1f)
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

        Dictionary<int, SpeakerFaceMapping> audioToFace = BuildAudioSpeakerToFaceMapping(endedAtUtc);
        List<SpeakerSegment> primarySegments = _audioSegments.Count > 0
            ? BuildAudioMappedSegments(audioToFace)
            : [.. _segments];

        ResolveUtterancesWithAudioMapping(audioToFace);
        ResolveEmotionSamplesWithAudioMapping(audioToFace);

        Dictionary<string, double> speaking = new(StringComparer.OrdinalIgnoreCase);
        foreach (SpeakerSegment seg in primarySegments)
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

        List<InterruptionEvent> interruptionEvents = ComputeInterruptionEvents(primarySegments, endedAtUtc);
        Dictionary<string, int> interruptions = new(StringComparer.OrdinalIgnoreCase);
        foreach (InterruptionEvent e in interruptionEvents)
        {
            _ = interruptions.TryGetValue(e.InterrupterSpeakerKey, out int c);
            interruptions[e.InterrupterSpeakerKey] = c + 1;
        }

        // Turn-taking (count segments per speaker)
        Dictionary<string, int> turns = new(StringComparer.OrdinalIgnoreCase);
        foreach (SpeakerSegment seg in primarySegments)
        {
            _ = turns.TryGetValue(seg.SpeakerKey, out int t);
            turns[seg.SpeakerKey] = t + 1;
        }

        // Conversation graph (who follows who)
        Dictionary<(string From, string To), int> edgeCounts = new();
        for (int i = 1; i < primarySegments.Count; i++)
        {
            SpeakerSegment prev = primarySegments[i - 1];
            SpeakerSegment cur = primarySegments[i];
            if (string.Equals(prev.SpeakerKey, cur.SpeakerKey, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            (string From, string To) key = (prev.SpeakerKey, cur.SpeakerKey);
            _ = edgeCounts.TryGetValue(key, out int c);
            edgeCounts[key] = c + 1;
        }

        List<ConversationEdge> graph = [.. edgeCounts
            .OrderByDescending(kvp => kvp.Value)
            .Select(kvp => new ConversationEdge { FromSpeakerKey = kvp.Key.From, ToSpeakerKey = kvp.Key.To, Count = kvp.Value })];

        // Participation score (transparent heuristic)
        const double wSpeak = 1.0;
        const double wTurns = 5.0;
        const double wInterrupt = 3.0;
        Dictionary<string, double> participation = new(StringComparer.OrdinalIgnoreCase);
        foreach (string spk in speaking.Keys.Union(turns.Keys, StringComparer.OrdinalIgnoreCase).Union(interruptions.Keys, StringComparer.OrdinalIgnoreCase))
        {
            _ = speaking.TryGetValue(spk, out double sec);
            _ = turns.TryGetValue(spk, out int t);
            _ = interruptions.TryGetValue(spk, out int intr);
            participation[spk] = (wSpeak * sec) + (wTurns * t) - (wInterrupt * intr);
        }

        (Dictionary<string, int> emotionOverall, Dictionary<string, Dictionary<string, int>> emotionBySpeaker) = ComputeEmotionAggregates();

        return new MeetingSession
        {
            StartedAtUtc = _startedAtUtc,
            EndedAtUtc = endedAtUtc,
            Segments = primarySegments,
            VisualSegments = [.. _segments],
            SpeakingTimeSecondsBySpeaker = speaking,
            InterruptionsBySpeaker = interruptions,
            InterruptionEvents = interruptionEvents,
            Utterances = [.. _utterances],
            AudioSpeakerSegments = [.. _audioSegments],
            AudioSpeakerToFace = audioToFace,
            TurnsBySpeaker = turns,
            ParticipationScoreBySpeaker = participation,
            ConversationGraph = graph,
            EmotionSamples = [.. _emotionSamples],
            EmotionCountsOverall = emotionOverall,
            EmotionCountsBySpeaker = emotionBySpeaker
        };
    }

    private static List<InterruptionEvent> ComputeInterruptionEvents(IReadOnlyList<SpeakerSegment> segments, DateTime endedAtUtc)
    {
        // Corporate-style heuristic: a speaker "interrupts" when they start speaking while another speaker is still speaking.
        // (Uses overlaps rather than gaps, tolerant to minor diarization jitter.)
        const double minOverlapSeconds = 0.25;
        TimeSpan tolerance = TimeSpan.FromMilliseconds(150);

        List<(DateTime Start, DateTime End, string Key)> segs = new();
        foreach (SpeakerSegment s in segments)
        {
            DateTime end = s.EndUtc ?? endedAtUtc;
            if (end <= s.StartUtc)
            {
                continue;
            }

            segs.Add((s.StartUtc, end, s.SpeakerKey));
        }

        segs.Sort((a, b) => a.Start.CompareTo(b.Start));

        List<InterruptionEvent> events = new();
        for (int i = 0; i < segs.Count; i++)
        {
            (DateTime Start, DateTime End, string Key) cur = segs[i];

            // Find the best overlapping "previous" segment from another speaker.
            int bestIdx = -1;
            double bestOverlap = 0;
            for (int j = i - 1; j >= 0; j--)
            {
                (DateTime Start, DateTime End, string Key) prev = segs[j];
                if (prev.End <= cur.Start + tolerance)
                {
                    break; // earlier segments will also end before cur starts
                }

                if (string.Equals(prev.Key, cur.Key, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                DateTime overlapEnd = prev.End < cur.End ? prev.End : cur.End;
                double overlap = (overlapEnd - cur.Start).TotalSeconds;
                if (overlap > bestOverlap)
                {
                    bestOverlap = overlap;
                    bestIdx = j;
                }
            }

            if (bestIdx >= 0 && bestOverlap >= minOverlapSeconds)
            {
                string interruptedKey = segs[bestIdx].Key;

                // De-dup rapid re-triggers (e.g., diarization micro-splits).
                InterruptionEvent? last = events.LastOrDefault(e =>
                    string.Equals(e.InterrupterSpeakerKey, cur.Key, StringComparison.OrdinalIgnoreCase) &&
                    string.Equals(e.InterruptedSpeakerKey, interruptedKey, StringComparison.OrdinalIgnoreCase));

                if (last == null || (cur.Start - last.WhenUtc) > TimeSpan.FromMilliseconds(800))
                {
                    events.Add(new InterruptionEvent
                    {
                        WhenUtc = cur.Start,
                        InterrupterSpeakerKey = cur.Key,
                        InterruptedSpeakerKey = interruptedKey,
                        OverlapSeconds = Math.Round(bestOverlap, 3)
                    });
                }
            }
        }

        return events;
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

        int? audioSpeakerId = FindBestOverlappingAudioSpeakerId(startUtc, endUtc);
        if (audioSpeakerId.HasValue)
        {
            _utterances.Add(new Utterance
            {
                StartUtc = startUtc,
                EndUtc = endUtc,
                SpeakerKey = $"Audio:{audioSpeakerId.Value}",
                DisplayName = null,
                Text = text.Trim(),
                TrackId = null,
                AudioSpeakerId = audioSpeakerId.Value
            });
            return;
        }

        SpeakerSegment? best = FindBestOverlappingSpeaker(startUtc, endUtc);
        _utterances.Add(new Utterance
        {
            StartUtc = startUtc,
            EndUtc = endUtc,
            SpeakerKey = best?.SpeakerKey ?? "Unknown",
            DisplayName = best?.DisplayName,
            Text = text.Trim(),
            TrackId = best?.TrackId,
            AudioSpeakerId = best?.AudioSpeakerId
        });
    }

    private int? FindBestOverlappingAudioSpeakerId(DateTime startUtc, DateTime endUtc)
    {
        int? bestId = null;
        double bestOverlap = 0;

        foreach (AudioSpeakerSegment seg in _audioSegments)
        {
            if (seg.EndUtc <= startUtc || seg.StartUtc >= endUtc)
            {
                continue;
            }

            DateTime overlapStart = seg.StartUtc > startUtc ? seg.StartUtc : startUtc;
            DateTime overlapEnd = seg.EndUtc < endUtc ? seg.EndUtc : endUtc;
            double overlap = (overlapEnd - overlapStart).TotalSeconds;
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                bestId = seg.SpeakerId;
            }
        }

        return bestId;
    }

    /// <summary>
    /// Aggregates and returns the top speakers by total duration.
    /// </summary>
    /// <param name="nowUtc">The cutoff boundary to measure active open segments.</param>
    /// <param name="top">Maximum length of the resulting list.</param>
    /// <returns>A collection of tuples containing speaker key, display name, and duration in seconds.</returns>
    public IReadOnlyList<(string SpeakerKey, string? DisplayName, double Seconds)> GetSpeakingTimeSoFar(DateTime nowUtc, int top = 3)
    {
        List<SpeakerSegment> primary = GetPrimarySegmentsForNow(nowUtc, out Dictionary<string, string?> names);
        Dictionary<string, double> totals = new(StringComparer.OrdinalIgnoreCase);

        foreach (SpeakerSegment seg in primary)
        {
            DateTime end = seg.EndUtc ?? nowUtc;
            double dur = (end - seg.StartUtc).TotalSeconds;
            if (dur <= 0) continue;
            _ = totals.TryGetValue(seg.SpeakerKey, out double cur);
            totals[seg.SpeakerKey] = cur + dur;
        }

        return [.. totals
            .OrderByDescending(kvp => kvp.Value)
            .Take(Math.Clamp(top, 1, 10))
            .Select(kvp => (kvp.Key, names.TryGetValue(kvp.Key, out string? n) ? n : null, kvp.Value))];
    }

    public IReadOnlyList<(string SpeakerKey, string? DisplayName, double Score)> GetParticipationSoFar(DateTime nowUtc, int top = 3)
    {
        List<SpeakerSegment> primary = GetPrimarySegmentsForNow(nowUtc, out Dictionary<string, string?> names);

        // Speaking seconds
        Dictionary<string, double> speaking = new(StringComparer.OrdinalIgnoreCase);
        foreach (SpeakerSegment seg in primary)
        {
            DateTime end = seg.EndUtc ?? nowUtc;
            double dur = (end - seg.StartUtc).TotalSeconds;
            if (dur <= 0) continue;
            _ = speaking.TryGetValue(seg.SpeakerKey, out double cur);
            speaking[seg.SpeakerKey] = cur + dur;
        }

        // Turns
        Dictionary<string, int> turns = new(StringComparer.OrdinalIgnoreCase);
        foreach (SpeakerSegment seg in primary)
        {
            _ = turns.TryGetValue(seg.SpeakerKey, out int t);
            turns[seg.SpeakerKey] = t + 1;
        }

        List<InterruptionEvent> ev = ComputeInterruptionEvents(primary, nowUtc);
        Dictionary<string, int> interruptions = new(StringComparer.OrdinalIgnoreCase);
        foreach (InterruptionEvent e in ev)
        {
            _ = interruptions.TryGetValue(e.InterrupterSpeakerKey, out int c);
            interruptions[e.InterrupterSpeakerKey] = c + 1;
        }

        const double wSpeak = 1.0;
        const double wTurns = 5.0;
        const double wInterrupt = 3.0;
        Dictionary<string, double> scores = new(StringComparer.OrdinalIgnoreCase);
        foreach (string spk in speaking.Keys.Union(turns.Keys, StringComparer.OrdinalIgnoreCase).Union(interruptions.Keys, StringComparer.OrdinalIgnoreCase))
        {
            _ = speaking.TryGetValue(spk, out double sec);
            _ = turns.TryGetValue(spk, out int t);
            _ = interruptions.TryGetValue(spk, out int intr);
            scores[spk] = (wSpeak * sec) + (wTurns * t) - (wInterrupt * intr);
        }

        return [.. scores
            .OrderByDescending(kvp => kvp.Value)
            .Take(Math.Clamp(top, 1, 10))
            .Select(kvp => (kvp.Key, names.TryGetValue(kvp.Key, out string? n) ? n : null, kvp.Value))];
    }

    private List<SpeakerSegment> GetPrimarySegmentsForNow(DateTime nowUtc, out Dictionary<string, string?> namesByKey)
    {
        namesByKey = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);

        if (_audioSegments.Count == 0)
        {
            foreach (SpeakerSegment s in _segments)
            {
                _ = namesByKey.TryAdd(s.SpeakerKey, s.DisplayName);
            }
            return [.. _segments];
        }

        Dictionary<int, SpeakerFaceMapping> map = BuildAudioSpeakerToFaceMapping(nowUtc);
        List<SpeakerSegment> primary = BuildAudioMappedSegments(map);
        foreach (SpeakerSegment s in primary)
        {
            _ = namesByKey.TryAdd(s.SpeakerKey, s.DisplayName);
        }

        return primary;
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
    /// Estimates which visual track corresponds to which audio cluster ID.
    /// </summary>
    /// <returns>A mapping from Audio Speaker ID to Face Track metadata.</returns>
    private Dictionary<int, SpeakerFaceMapping> BuildAudioSpeakerToFaceMapping(DateTime endedAtUtc)
    {
        // Prefer overlap between diarization segments and visual speaker segments; fall back to observed co-occurrence.
        Dictionary<(int SpeakerId, int TrackId), double> weights = new();

        foreach (AudioSpeakerSegment a in _audioSegments)
        {
            foreach (SpeakerSegment v in _segments)
            {
                if (!v.TrackId.HasValue)
                {
                    continue;
                }

                DateTime vEnd = v.EndUtc ?? endedAtUtc;
                if (vEnd <= a.StartUtc || v.StartUtc >= a.EndUtc)
                {
                    continue;
                }

                DateTime overlapStart = v.StartUtc > a.StartUtc ? v.StartUtc : a.StartUtc;
                DateTime overlapEnd = vEnd < a.EndUtc ? vEnd : a.EndUtc;
                double sec = (overlapEnd - overlapStart).TotalSeconds;
                if (sec <= 0)
                {
                    continue;
                }

                (int SpeakerId, int TrackId) k = (a.SpeakerId, v.TrackId.Value);
                _ = weights.TryGetValue(k, out double cur);
                weights[k] = cur + sec;
            }
        }

        // Fallback: co-occurrence counts (small weight) for cases where overlap isn't available.
        foreach (KeyValuePair<(int SpeakerId, int TrackId), int> kvp in _cooccurrence)
        {
            if (!weights.ContainsKey(kvp.Key))
            {
                weights[kvp.Key] = kvp.Value * 0.05;
            }
        }

        Dictionary<int, List<KeyValuePair<(int SpeakerId, int TrackId), double>>> bySpeaker = weights
            .GroupBy(kvp => kvp.Key.SpeakerId)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(x => x.Value).ToList());

        HashSet<int> usedTracks = new();
        Dictionary<int, SpeakerFaceMapping> map = new();

        foreach (int speakerId in bySpeaker.Keys.OrderBy(k => k))
        {
            List<KeyValuePair<(int SpeakerId, int TrackId), double>> candidates = bySpeaker[speakerId];
            KeyValuePair<(int SpeakerId, int TrackId), double> best = candidates.FirstOrDefault(c => !usedTracks.Contains(c.Key.TrackId));
            if (best.Key.TrackId > 0)
            {
                _ = usedTracks.Add(best.Key.TrackId);
                string? display = _segments.LastOrDefault(s => s.TrackId == best.Key.TrackId && !string.IsNullOrWhiteSpace(s.DisplayName))?.DisplayName;
                map[speakerId] = new SpeakerFaceMapping
                {
                    TrackId = best.Key.TrackId,
                    DisplayName = display,
                    Score = (float)best.Value
                };
            }
            else
            {
                map[speakerId] = new SpeakerFaceMapping { TrackId = null, DisplayName = null, Score = 0 };
            }
        }

        return map;
    }

    private void ResolveUtterancesWithAudioMapping(Dictionary<int, SpeakerFaceMapping> audioToFace)
    {
        for (int i = 0; i < _utterances.Count; i++)
        {
            Utterance u = _utterances[i];
            int? spkId = u.AudioSpeakerId;
            if (spkId == null && u.SpeakerKey.StartsWith("Audio:", StringComparison.OrdinalIgnoreCase))
            {
                string raw = u.SpeakerKey["Audio:".Length..].Trim();
                if (int.TryParse(raw, out int parsed) && parsed > 0)
                {
                    spkId = parsed;
                }
            }

            if (spkId.HasValue &&
                audioToFace.TryGetValue(spkId.Value, out SpeakerFaceMapping map) &&
                map.TrackId.HasValue)
            {
                int trackId = map.TrackId.Value;
                string key = !string.IsNullOrWhiteSpace(map.DisplayName) ? $"Name:{map.DisplayName}" : $"Track:{trackId}";
                _utterances[i] = new Utterance
                {
                    StartUtc = u.StartUtc,
                    EndUtc = u.EndUtc,
                    SpeakerKey = key,
                    DisplayName = map.DisplayName,
                    Text = u.Text,
                    TrackId = trackId,
                    AudioSpeakerId = spkId.Value
                };
            }
        }
    }

    private void ResolveEmotionSamplesWithAudioMapping(Dictionary<int, SpeakerFaceMapping> audioToFace)
    {
        // If we have a mapping to named tracks, normalize emotion samples to match the final speaker keys where possible.
        for (int i = 0; i < _emotionSamples.Count; i++)
        {
            EmotionSample s = _emotionSamples[i];
            if (s.TrackId == null)
            {
                continue;
            }

            // If any audio speaker maps to this track, prefer the mapped display name for consistent reporting.
            SpeakerFaceMapping? map = audioToFace.Values.FirstOrDefault(v => v.TrackId == s.TrackId);
            if (map?.TrackId == null)
            {
                continue;
            }

            string? display = NormalizeDisplayName(map.DisplayName) ?? s.DisplayName;
            string key = !string.IsNullOrWhiteSpace(display) ? $"Name:{display}" : $"Track:{map.TrackId.Value}";

            if (!string.Equals(key, s.SpeakerKey, StringComparison.OrdinalIgnoreCase) || s.DisplayName != display)
            {
                _emotionSamples[i] = new EmotionSample
                {
                    WhenUtc = s.WhenUtc,
                    SpeakerKey = key,
                    DisplayName = display,
                    TrackId = map.TrackId,
                    Emotion = s.Emotion,
                    Confidence = s.Confidence
                };
            }
        }
    }

    private (Dictionary<string, int> Overall, Dictionary<string, Dictionary<string, int>> BySpeaker) ComputeEmotionAggregates()
    {
        Dictionary<string, int> overall = new(StringComparer.OrdinalIgnoreCase);
        Dictionary<string, Dictionary<string, int>> bySpeaker = new(StringComparer.OrdinalIgnoreCase);

        foreach (EmotionSample s in _emotionSamples)
        {
            if (string.IsNullOrWhiteSpace(s.Emotion))
            {
                continue;
            }

            _ = overall.TryGetValue(s.Emotion, out int o);
            overall[s.Emotion] = o + 1;

            if (!bySpeaker.TryGetValue(s.SpeakerKey, out Dictionary<string, int>? map))
            {
                map = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                bySpeaker[s.SpeakerKey] = map;
            }

            _ = map.TryGetValue(s.Emotion, out int c);
            map[s.Emotion] = c + 1;
        }

        return (overall, bySpeaker);
    }

    private List<SpeakerSegment> BuildAudioMappedSegments(Dictionary<int, SpeakerFaceMapping> audioToFace)
    {
        List<SpeakerSegment> mapped = new();
        foreach (AudioSpeakerSegment a in _audioSegments.OrderBy(s => s.StartUtc))
        {
            SpeakerFaceMapping? m = audioToFace.TryGetValue(a.SpeakerId, out SpeakerFaceMapping mm) ? mm : null;
            string? display = m?.DisplayName;
            int? trackId = m?.TrackId;
            string key = trackId.HasValue
                ? (!string.IsNullOrWhiteSpace(display) ? $"Name:{display}" : $"Track:{trackId.Value}")
                : $"Audio:{a.SpeakerId}";

            mapped.Add(new SpeakerSegment
            {
                StartUtc = a.StartUtc,
                EndUtc = a.EndUtc,
                SpeakerKey = key,
                DisplayName = display,
                TrackId = trackId,
                AudioSpeakerId = a.SpeakerId
            });
        }

        // Merge adjacent segments with the same speaker key.
        mapped.Sort((x, y) => x.StartUtc.CompareTo(y.StartUtc));
        List<SpeakerSegment> merged = new();
        foreach (SpeakerSegment seg in mapped)
        {
            SpeakerSegment? last = merged.LastOrDefault();
            if (last != null &&
                string.Equals(last.SpeakerKey, seg.SpeakerKey, StringComparison.OrdinalIgnoreCase) &&
                last.EndUtc.HasValue &&
                seg.EndUtc.HasValue &&
                seg.StartUtc <= last.EndUtc.Value + TimeSpan.FromMilliseconds(200))
            {
                last.EndUtc = seg.EndUtc > last.EndUtc ? seg.EndUtc : last.EndUtc;
                continue;
            }

            merged.Add(seg);
        }

        return merged;
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
