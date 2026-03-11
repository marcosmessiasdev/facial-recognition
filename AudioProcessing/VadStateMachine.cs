namespace AudioProcessing;

/// <summary>
/// Debounces raw VAD probabilities into a stable speech on/off signal using timing constraints.
/// </summary>
public sealed class VadStateMachine(int minSpeechMs = 120, int minSilenceMs = 120, int hangoverMs = 350)
{
    private readonly TimeSpan _minSpeech = TimeSpan.FromMilliseconds(Math.Clamp(minSpeechMs, 0, 2000));
    private readonly TimeSpan _minSilence = TimeSpan.FromMilliseconds(Math.Clamp(minSilenceMs, 0, 2000));
    private readonly TimeSpan _hangover = TimeSpan.FromMilliseconds(Math.Clamp(hangoverMs, 0, 5000));

    private DateTime? _aboveSinceUtc;
    private DateTime? _belowSinceUtc;
    private DateTime _lastAboveUtc;

    public bool IsSpeechActive { get; private set; }

    public void Reset()
    {
        IsSpeechActive = false;
        _aboveSinceUtc = null;
        _belowSinceUtc = null;
        _lastAboveUtc = default;
    }

    /// <summary>
    /// Updates the internal state and returns the stable speech activity flag.
    /// </summary>
    /// <param name="nowUtc">Current UTC timestamp for consistent time tracking.</param>
    /// <param name="prob">Raw speech probability/score from the model.</param>
    /// <param name="threshold">Threshold above which the model is considered "speech present".</param>
    public bool Update(DateTime nowUtc, float prob, float threshold)
    {
        bool above = prob >= threshold;

        if (above)
        {
            _lastAboveUtc = nowUtc;
            _belowSinceUtc = null;
            _aboveSinceUtc ??= nowUtc;

            if (!IsSpeechActive)
            {
                if (_minSpeech == TimeSpan.Zero || (nowUtc - _aboveSinceUtc.Value) >= _minSpeech)
                {
                    IsSpeechActive = true;
                }
            }

            return IsSpeechActive;
        }

        // below threshold
        _aboveSinceUtc = null;
        _belowSinceUtc ??= nowUtc;

        if (IsSpeechActive)
        {
            bool hangoverExpired = _hangover == TimeSpan.Zero || (nowUtc - _lastAboveUtc) >= _hangover;
            bool minSilenceMet = _minSilence == TimeSpan.Zero || (nowUtc - _belowSinceUtc.Value) >= _minSilence;
            if (hangoverExpired && minSilenceMet)
            {
                IsSpeechActive = false;
            }
        }

        return IsSpeechActive;
    }
}

