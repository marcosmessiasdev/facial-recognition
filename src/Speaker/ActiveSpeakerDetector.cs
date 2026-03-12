using FacialRecognition.Domain;

namespace SpeakerDetection;

/// <summary>
/// Logic component that identifies the most likely active speaker among tracked faces.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Determines identity of the current speaker by combining audio speech status 
/// with visual mouth motion cues.
///
/// Responsibilities:
/// - Order tracked individuals by their mouth motion intensity.
/// - Filter candidates based on a minimum visual motion threshold.
/// - Implement a persistence "hold" period to prevent rapid flickering of the speaker status.
/// - Return the stable ID of the currently active speaker.
///
/// Dependencies:
/// - FaceTracking (Track metadata)
///
/// Architectural Role:
/// Infrastructure Component / Logic Service.
///
/// Constraints:
/// - Relies on external VAD status; if speech is not detected, it effectively clears the active speaker state.
/// </remarks>
public sealed class ActiveSpeakerDetector
{
    private readonly float _minMouthScore;
    private readonly float _minMouthScoreVisualOnly;
    private readonly float _minTalkNetProb;
    private readonly TimeSpan _holdTime;

    private int? _activeTrackId;
    private DateTime _activeUntilUtc;

    /// <summary>
    /// Initializes a new instance of the ActiveSpeakerDetector class.
    /// </summary>
    /// <param name="minMouthScore">The minimum motion score required to consider someone as speaking.</param>
    /// <param name="minMouthScoreVisualOnly">A higher threshold used when audio VAD is unavailable or inactive.</param>
    /// <param name="holdMs">Duration (in milliseconds) to stay active after a speaker stops moving their mouth.</param>
    public ActiveSpeakerDetector(float minMouthScore = 0.015f, float minMouthScoreVisualOnly = 0.03f, float minTalkNetProb = 0.55f, int holdMs = 400)
    {
        _minMouthScore = Math.Max(0f, minMouthScore);
        _minMouthScoreVisualOnly = Math.Max(_minMouthScore, minMouthScoreVisualOnly);
        _minTalkNetProb = Math.Clamp(minTalkNetProb, 0.05f, 0.99f);
        _holdTime = TimeSpan.FromMilliseconds(Math.Max(0, holdMs));
    }

    /// <summary>
    /// Evaluates tracked individuals and returns the ID of the person currently speaking.
    /// </summary>
    /// <param name="tracks">The list of all currently tracked faces.</param>
    /// <param name="speechActive">True if any speech is detected in the selected audio source.</param>
    /// <param name="allowVisualOnlyFallback">If true, selects a speaker using only mouth motion when speechActive is false.</param>
    /// <param name="nowUtc">The current UTC time.</param>
    /// <returns>The ID of the active speaker, or null if nobody is speaking.</returns>
    public int? Update(IReadOnlyList<Track> tracks, bool speechActive, bool allowVisualOnlyFallback, DateTime nowUtc)
    {
        ArgumentNullException.ThrowIfNull(tracks);

        foreach (Track t in tracks)
        {
            t.SpeakingScore = 0f;
        }

        if (!speechActive && !allowVisualOnlyFallback)
        {
            if (_activeTrackId != null && nowUtc <= _activeUntilUtc)
            {
                return _activeTrackId;
            }

            _activeTrackId = null;
            return null;
        }

        bool hasTalkNet = tracks.Any(t => t.TalkNetSpeakingProb > 1e-6f);
        if (hasTalkNet)
        {
            float totalProb = tracks.Sum(t => t.TalkNetSpeakingProb);
            if (totalProb > 1e-6f)
            {
                foreach (Track t in tracks)
                {
                    t.SpeakingScore = t.TalkNetSpeakingProb / totalProb;
                }
            }

            Track? bestProb = tracks
                .Where(t => t.TalkNetSpeakingProb >= _minTalkNetProb)
                .OrderByDescending(t => t.TalkNetSpeakingProb)
                .FirstOrDefault();

            if (bestProb == null)
            {
                if (_activeTrackId != null && nowUtc <= _activeUntilUtc)
                {
                    return _activeTrackId;
                }

                _activeTrackId = null;
                return null;
            }

            _activeTrackId = bestProb.Id;
            _activeUntilUtc = nowUtc + _holdTime;
            return _activeTrackId;
        }

        float min = speechActive ? _minMouthScore : _minMouthScoreVisualOnly;

        float total = tracks.Sum(t => t.MouthMotionScore);
        if (total > 1e-6f)
        {
            foreach (Track t in tracks)
            {
                t.SpeakingScore = t.MouthMotionScore / total;
            }
        }

        Track? best = tracks
            .Where(t => t.MouthMotionScore >= min)
            .OrderByDescending(t => t.MouthMotionScore)
            .FirstOrDefault();

        if (best == null)
        {
            // Keep previous speaker briefly to reduce flicker.
            if (_activeTrackId != null && nowUtc <= _activeUntilUtc)
            {
                return _activeTrackId;
            }

            _activeTrackId = null;
            return null;
        }

        _activeTrackId = best.Id;
        _activeUntilUtc = nowUtc + _holdTime;
        return _activeTrackId;
    }
}
