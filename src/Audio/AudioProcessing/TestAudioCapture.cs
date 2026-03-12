using System.Diagnostics;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AudioProcessing;

/// <summary>
/// Deterministic audio source for offline/E2E testing.
/// Reads a WAV file and emits fixed-size mono frames with stable timestamps.
/// </summary>
public sealed class TestAudioCapture : IAudioCapture
{
    private readonly object _sync = new();
    private readonly string _filePath;
    private readonly bool _loop;
    private readonly TimeSpan _initialSilence;

    private CancellationTokenSource? _cts;
    private Task? _pumpTask;
    private Stopwatch? _sw;
    private ISampleProvider? _pipeline;
    private WaveStream? _reader;

    public event EventHandler<AudioFrameEventArgs>? FrameArrived;

    public int TargetSampleRateHz { get; }
    public int FrameSizeSamples { get; }

    public string? SelectedDeviceName => "TestAudio";

    public TestAudioCapture(
        string filePath,
        bool loop = true,
        int initialSilenceMs = 0,
        int targetSampleRateHz = 16000,
        int frameSizeSamples = 512)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targetSampleRateHz);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frameSizeSamples);

        _filePath = string.IsNullOrWhiteSpace(filePath) ? throw new ArgumentException("Audio test file path is required.", nameof(filePath)) : filePath;
        _loop = loop;
        _initialSilence = TimeSpan.FromMilliseconds(Math.Clamp(initialSilenceMs, 0, 30_000));

        TargetSampleRateHz = targetSampleRateHz;
        FrameSizeSamples = frameSizeSamples;
    }

    public void Start()
    {
        lock (_sync)
        {
            if (_cts != null)
            {
                return;
            }

            if (!File.Exists(_filePath))
            {
                throw new FileNotFoundException("Test audio file not found.", _filePath);
            }

            _reader = new WaveFileReader(_filePath);
            ISampleProvider sample = _reader.ToSampleProvider();

            if (sample.WaveFormat.Channels == 2)
            {
                sample = new StereoToMonoSampleProvider(sample) { LeftVolume = 0.5f, RightVolume = 0.5f };
            }
            else if (sample.WaveFormat.Channels > 2)
            {
                sample = new MultiplexingSampleProvider([sample], 1);
            }

            if (sample.WaveFormat.SampleRate != TargetSampleRateHz)
            {
                sample = new WdlResamplingSampleProvider(sample, TargetSampleRateHz);
            }

            _pipeline = sample;
            _cts = new CancellationTokenSource();
            _sw = Stopwatch.StartNew();
            _pumpTask = Task.Run(() => PumpAsync(_cts.Token), _cts.Token);
        }
    }

    public void StopCapture()
    {
        lock (_sync)
        {
            _cts?.Cancel();
        }
    }

    // Back-compat
    public void Stop() => StopCapture();

    private async Task PumpAsync(CancellationToken ct)
    {
        float[] buffer = new float[FrameSizeSamples];
        TimeSpan offset = TimeSpan.Zero;

        if (_initialSilence > TimeSpan.Zero)
        {
            int totalSilentFrames = (int)Math.Ceiling(_initialSilence.TotalSeconds * TargetSampleRateHz / FrameSizeSamples);
            for (int i = 0; i < totalSilentFrames && !ct.IsCancellationRequested; i++)
            {
                Array.Clear(buffer, 0, buffer.Length);
                Emit(buffer, offset);
                offset += TimeSpan.FromSeconds((double)FrameSizeSamples / TargetSampleRateHz);
                await Task.Delay(TimeSpan.FromSeconds((double)FrameSizeSamples / TargetSampleRateHz), ct);
            }
        }

        while (!ct.IsCancellationRequested)
        {
            ISampleProvider? pipeline;
            lock (_sync)
            {
                pipeline = _pipeline;
            }

            if (pipeline == null)
            {
                await Task.Delay(50, ct);
                continue;
            }

            int read = pipeline.Read(buffer, 0, buffer.Length);
            if (read < buffer.Length)
            {
                if (_loop)
                {
                    lock (_sync)
                    {
                        if (_reader != null)
                        {
                            _reader.Position = 0;
                        }
                    }
                    continue;
                }

                break;
            }

            Emit(buffer, offset);
            offset += TimeSpan.FromSeconds((double)FrameSizeSamples / TargetSampleRateHz);
            await Task.Delay(TimeSpan.FromSeconds((double)FrameSizeSamples / TargetSampleRateHz), ct);
        }
    }

    private void Emit(float[] frame, TimeSpan offset)
    {
        float[] samples = new float[frame.Length];
        Array.Copy(frame, samples, frame.Length);
        FrameArrived?.Invoke(this, new AudioFrameEventArgs(samples, TargetSampleRateHz, offset));
    }

    public void Dispose()
    {
        lock (_sync)
        {
            _cts?.Cancel();
        }

        try { _ = (_pumpTask?.Wait(TimeSpan.FromSeconds(1))); } catch { /* ignore */ }

        lock (_sync)
        {
            _cts?.Dispose();
            _cts = null;

            _pipeline = null;
            _reader?.Dispose();
            _reader = null;

            _sw?.Stop();
            _sw = null;
        }

        GC.SuppressFinalize(this);
    }
}

