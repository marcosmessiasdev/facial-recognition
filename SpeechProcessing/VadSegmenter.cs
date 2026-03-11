using System.Collections.Concurrent;

namespace SpeechProcessing;

/// <summary>
/// Collects 16kHz mono PCM frames into speech segments using a boolean VAD signal.
/// </summary>
public sealed class VadSegmenter(int sampleRateHz = 16000, int hangoverMs = 350, int maxSegmentSeconds = 15) : IDisposable
{
    private readonly object _sync = new();
    private readonly int _sampleRateHz = sampleRateHz;
    private readonly TimeSpan _hangover = TimeSpan.FromMilliseconds(Math.Clamp(hangoverMs, 0, 2000));
    private readonly TimeSpan _maxSegment = TimeSpan.FromSeconds(Math.Clamp(maxSegmentSeconds, 2, 60));

    private readonly ConcurrentQueue<(float[] Samples, TimeSpan Time)> _queue = new();
    private CancellationTokenSource? _cts;
    private Task? _pumpTask;

    private bool _inSpeech;
    private TimeSpan _segmentStart;
    private TimeSpan _lastSpeechTime;
    private List<float>? _segment;

    public event EventHandler<(TimeSpan Start, TimeSpan End, float[] Samples)>? SegmentReady;

    public void Start()
    {
        lock (_sync)
        {
            if (_cts != null)
            {
                return;
            }

            _cts = new CancellationTokenSource();
            _pumpTask = Task.Run(() => PumpAsync(_cts.Token), _cts.Token);
        }
    }

    public void Stop()
    {
        lock (_sync)
        {
            _cts?.Cancel();
        }

        try { _ = (_pumpTask?.Wait(TimeSpan.FromSeconds(1))); } catch { /* ignore */ }
    }

    public void Push(float[] samples, TimeSpan timestamp, bool speechActive)
    {
        ArgumentNullException.ThrowIfNull(samples);

        // Keep this lock-free for the capture thread.
        _queue.Enqueue((samples, timestamp));

        // Update VAD state (cheap atomic-like update).
        if (speechActive)
        {
            _inSpeech = true;
            _lastSpeechTime = timestamp;
            if (_segment == null)
            {
                _segmentStart = timestamp;
                _segment = new List<float>(samples.Length * 16);
            }
        }
    }

    private async Task PumpAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            if (!_queue.TryDequeue(out (float[] Samples, TimeSpan Time) item))
            {
                await Task.Delay(10, ct);
                continue;
            }

            (float[]? samples, TimeSpan time) = item;
            List<float>? seg = _segment;
            if (seg != null)
            {
                seg.AddRange(samples);

                TimeSpan segDur = time - _segmentStart;
                TimeSpan silentFor = time - _lastSpeechTime;

                if (segDur >= _maxSegment || (silentFor >= _hangover && _inSpeech))
                {
                    // End segment.
                    _inSpeech = false;
                    _segment = null;

                    float[] pcm = seg.ToArray();
                    SegmentReady?.Invoke(this, (_segmentStart, time, pcm));
                }
            }
        }
    }

    public void Dispose()
    {
        Stop();
        _cts?.Dispose();
        _cts = null;
    }
}
