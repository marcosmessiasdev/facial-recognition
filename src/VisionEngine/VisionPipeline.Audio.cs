using AudioProcessing;
using Logging;
using SpeechProcessing;

namespace VisionEngine;

public partial class VisionPipeline
{
    /// <summary>
    /// Handles incoming audio frames and performs Voice Activity Detection (VAD).
    /// </summary>
    private void OnAudioFrameArrived(object? sender, AudioFrameEventArgs e)
    {
        if (!_acceptAudioFrames)
        {
            return;
        }

        SileroVad? vad = _vad;
        if (vad == null)
        {
            return;
        }

        // Heuristic gain for very quiet loopback audio (common when capturing meeting audio).
        // This improves VAD/STT robustness across devices without needing per-machine tuning.
        float maxAbsFrame = 0f;
        for (int i = 0; i < e.Samples.Length; i++)
        {
            float a = MathF.Abs(e.Samples[i]);
            if (a > maxAbsFrame)
            {
                maxAbsFrame = a;
            }
        }

        if (maxAbsFrame is > 0f and < 0.05f)
        {
            float gain = MathF.Min(20f, 0.2f / maxAbsFrame);
            for (int i = 0; i < e.Samples.Length; i++)
            {
                float v = e.Samples[i] * gain;
                e.Samples[i] = Math.Clamp(v, -1f, 1f);
            }
        }

        // Lightweight audio stats to help diagnose routing/silence issues.
        DateTime now = DateTime.UtcNow;
        _lastAudioOffset = e.Offset;
        if ((now - _lastAudioStatsLogUtc) > TimeSpan.FromSeconds(2))
        {
            _lastAudioStatsLogUtc = now;

            float sumSq = 0f;
            float maxAbs = 0f;
            for (int i = 0; i < e.Samples.Length; i++)
            {
                float s = e.Samples[i];
                float a = MathF.Abs(s);
                if (a > maxAbs)
                {
                    maxAbs = a;
                }

                sumSq += s * s;
            }

            float rms = e.Samples.Length > 0 ? MathF.Sqrt(sumSq / e.Samples.Length) : 0f;
            AppLogger.Instance.Debug("Audio rms={Rms:0.000} max={Max:0.000} sr={Sr}", rms, maxAbs, e.SampleRateHz);
        }

        float prob;
        try
        {
            prob = vad.GetSpeechProbability(e.Samples, e.SampleRateHz);
        }
        catch (Exception ex)
        {
            AppLogger.Instance.Warning(ex, "VAD inference failed");
            return;
        }

        VadStateMachine? sm = _vadStateMachine;
        bool active = sm != null
            ? sm.Update(now, prob, _cfg.AudioVadSpeechThreshold)
            : prob >= _cfg.AudioVadSpeechThreshold;
        bool stateChanged = active != _vadSpeechActive;
        _vadSpeechActive = active;
        _vadSpeechProb = prob;

        _audioBaseUtc ??= now - e.Offset;
        if (stateChanged || (now - _lastVadLogUtc) > TimeSpan.FromSeconds(2))
        {
            _lastVadLogUtc = now;
            AppLogger.Instance.Debug("VAD speech={Speech} prob={Prob:0.000}", _vadSpeechActive, prob);
        }

        try
        {
            _stt?.PushFrame(e.Samples, e.Offset, active);
        }
        catch
        {
            // ignore
        }

        try
        {
            _diarizer?.PushFrame(e.Samples, e.Offset, active);
        }
        catch
        {
            // ignore
        }

        lock (_audioChunksSync)
        {
            _audioChunks.Add((e.Offset, e.Samples));
            TimeSpan cutoff = e.Offset - TimeSpan.FromSeconds(6);
            int remove = 0;
            while (remove < _audioChunks.Count && _audioChunks[remove].Offset < cutoff)
            {
                remove++;
            }

            if (remove > 0)
            {
                _audioChunks.RemoveRange(0, remove);
            }
        }
    }

    private void OnTranscriptReady(object? sender, TranscriptSegment seg)
    {
        _lastTranscript = seg.Text;
        AppLogger.Instance.Information("Transcript [{Start}-{End}]: {Text}", seg.Start, seg.End, seg.Text);

        try
        {
            if (_analytics != null)
            {
                DateTime baseUtc = _audioBaseUtc ?? DateTime.UtcNow;
                DateTime startUtc = baseUtc + seg.Start;
                DateTime endUtc = baseUtc + seg.End;
                _analytics.AddUtterance(startUtc, endUtc, seg.Text);
            }
        }
        catch
        {
            // ignore
        }
    }

    private void OnTranscriptionFailed(object? sender, Exception ex)
    {
        AppLogger.Instance.Debug(ex, "Transcription failed (best-effort)");
    }

    private void OnSpeechSegmentReady(object? sender, (TimeSpan Start, TimeSpan End, int SampleCount) seg)
    {
        double seconds = (seg.End - seg.Start).TotalSeconds;
        AppLogger.Instance.Information("Speech segment [{Start}-{End}] dur={Dur:0.00}s samples={Samples}",
            seg.Start, seg.End, seconds, seg.SampleCount);
    }

    private void OnAudioSpeakerSegmentReady(object? sender, SpeakerDiarization.AudioSpeakerSegment seg)
    {
        _activeAudioSpeakerId = seg.SpeakerId;
        _activeAudioSpeakerConfidence = seg.Confidence;
        AppLogger.Instance.Information("Diarization segment speaker={Speaker} [{Start}-{End}] conf={Conf:0.00}",
            seg.SpeakerId, seg.Start, seg.End, seg.Confidence);

        try
        {
            if (_analytics != null)
            {
                DateTime baseUtc = _audioBaseUtc ?? DateTime.UtcNow;
                _analytics.AddAudioSpeakerSegment(baseUtc + seg.Start, baseUtc + seg.End, seg.SpeakerId, seg.Confidence);
            }
        }
        catch
        {
            // ignore
        }
    }

    private void OnActiveAudioSpeakerUpdated(object? sender, (TimeSpan Offset, int SpeakerId, float Confidence) e)
    {
        if (e.SpeakerId <= 0)
        {
            return;
        }

        _activeAudioSpeakerId = e.SpeakerId;
        _activeAudioSpeakerConfidence = e.Confidence;
    }

    private bool TryGetRecentAudioWindow(TimeSpan endOffset, TimeSpan window, out float[] audio)
    {
        audio = [];

        int sr = _cfg.AudioVadSampleRateHz;
        int needed = (int)Math.Ceiling(window.TotalSeconds * sr);
        if (needed <= 0)
        {
            return false;
        }

        List<float> samples = new();
        lock (_audioChunksSync)
        {
            // Gather chunks up to endOffset.
            foreach ((TimeSpan Offset, float[]? Samples) in _audioChunks)
            {
                if (Offset > endOffset)
                {
                    break;
                }

                samples.AddRange(Samples);
            }
        }

        if (samples.Count < needed)
        {
            return false;
        }

        audio = [.. samples.Skip(samples.Count - needed).Take(needed)];
        return true;
    }
}
