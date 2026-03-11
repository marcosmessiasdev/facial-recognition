using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace SpeechProcessing;

/// <summary>
/// Collects 16kHz mono PCM frames into speech segments using a boolean VAD signal.
/// </summary>
public sealed class VadSegmenter : IDisposable
{
    private readonly object _sync = new();
    private readonly int _sampleRateHz;
    private readonly TimeSpan _hangover;
    private readonly TimeSpan _maxSegment;

    private readonly ConcurrentQueue<(float[] Samples, TimeSpan Time)> _queue = new();
    private CancellationTokenSource? _cts;
    private Task? _pumpTask;

    private bool _inSpeech;
    private TimeSpan _segmentStart;
    private TimeSpan _lastSpeechTime;
    private List<float>? _segment;

    public event EventHandler<(TimeSpan Start, TimeSpan End, float[] Samples)>? SegmentReady;

    public VadSegmenter(int sampleRateHz = 16000, int hangoverMs = 350, int maxSegmentSeconds = 15)
    {
        _sampleRateHz = sampleRateHz;
        _hangover = TimeSpan.FromMilliseconds(Math.Clamp(hangoverMs, 0, 2000));
        _maxSegment = TimeSpan.FromSeconds(Math.Clamp(maxSegmentSeconds, 2, 60));
    }

    public void Start()
    {
        lock (_sync)
        {
            if (_cts != null) return;
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

        try { _pumpTask?.Wait(TimeSpan.FromSeconds(1)); } catch { /* ignore */ }
    }

    public void Push(float[] samples, TimeSpan timestamp, bool speechActive)
    {
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
            if (!_queue.TryDequeue(out var item))
            {
                await Task.Delay(10, ct);
                continue;
            }

            var (samples, time) = item;
            var seg = _segment;
            if (seg != null)
            {
                seg.AddRange(samples);

                var segDur = time - _segmentStart;
                var silentFor = time - _lastSpeechTime;

                if (segDur >= _maxSegment || (silentFor >= _hangover && _inSpeech))
                {
                    // End segment.
                    _inSpeech = false;
                    _segment = null;

                    var pcm = seg.ToArray();
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

