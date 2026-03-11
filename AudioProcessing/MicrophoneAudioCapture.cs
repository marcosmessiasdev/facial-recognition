using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AudioProcessing;

/// <summary>
/// Captures microphone audio (WASAPI capture) and emits resampled mono frames suitable for VAD/STT.
/// </summary>
public sealed class MicrophoneAudioCapture : IAudioCapture
{
    private readonly object _sync = new();
    private readonly string? _deviceSelector;

    private WasapiCapture? _capture;
    private BufferedWaveProvider? _buffered;
    private ISampleProvider? _pipeline;
    private CancellationTokenSource? _cts;
    private Task? _pumpTask;
    private Stopwatch? _sw;

    public event EventHandler<AudioFrameEventArgs>? FrameArrived;

    public int TargetSampleRateHz { get; }
    public int FrameSizeSamples { get; }

    public string? SelectedDeviceName { get; private set; }

    public MicrophoneAudioCapture(int targetSampleRateHz = 16000, int frameSizeSamples = 512, string? deviceSelector = null)
    {
        if (targetSampleRateHz <= 0) throw new ArgumentOutOfRangeException(nameof(targetSampleRateHz));
        if (frameSizeSamples <= 0) throw new ArgumentOutOfRangeException(nameof(frameSizeSamples));
        TargetSampleRateHz = targetSampleRateHz;
        FrameSizeSamples = frameSizeSamples;
        _deviceSelector = string.IsNullOrWhiteSpace(deviceSelector) ? null : deviceSelector.Trim();
    }

    public void Start()
    {
        lock (_sync)
        {
            if (_capture != null) return;

            var device = ResolveDevice(_deviceSelector);
            SelectedDeviceName = device?.FriendlyName;
            _capture = device != null ? new WasapiCapture(device) : new WasapiCapture();
            _buffered = new BufferedWaveProvider(_capture.WaveFormat)
            {
                DiscardOnBufferOverflow = true
            };

            _capture.DataAvailable += OnDataAvailable;
            _capture.RecordingStopped += OnStopped;

            var sample = _buffered.ToSampleProvider();

            if (sample.WaveFormat.Channels == 2)
            {
                sample = new StereoToMonoSampleProvider(sample)
                {
                    LeftVolume = 0.5f,
                    RightVolume = 0.5f
                };
            }
            else if (sample.WaveFormat.Channels > 2)
            {
                sample = new MultiplexingSampleProvider(new[] { sample }, 1);
            }

            if (sample.WaveFormat.SampleRate != TargetSampleRateHz)
            {
                sample = new WdlResamplingSampleProvider(sample, TargetSampleRateHz);
            }

            _pipeline = sample;
            _cts = new CancellationTokenSource();
            _sw = Stopwatch.StartNew();
            _pumpTask = Task.Run(() => PumpAsync(_cts.Token), _cts.Token);

            _capture.StartRecording();
        }
    }

    private static MMDevice? ResolveDevice(string? selector)
    {
        try
        {
            using var enumerator = new MMDeviceEnumerator();
            var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Capture, DeviceState.Active);

            if (!string.IsNullOrWhiteSpace(selector))
            {
                var byId = devices.FirstOrDefault(d => string.Equals(d.ID, selector, StringComparison.OrdinalIgnoreCase));
                if (byId != null) return byId;

                var byName = devices.FirstOrDefault(d => d.FriendlyName.Contains(selector, StringComparison.OrdinalIgnoreCase));
                if (byName != null) return byName;
            }

            return enumerator.GetDefaultAudioEndpoint(DataFlow.Capture, Role.Multimedia);
        }
        catch
        {
            return null;
        }
    }

    public void Stop()
    {
        lock (_sync)
        {
            if (_capture == null) return;
            try { _capture.StopRecording(); } catch { /* ignore */ }
        }
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        var buffered = _buffered;
        if (buffered == null) return;

        try
        {
            buffered.AddSamples(e.Buffer, 0, e.BytesRecorded);
        }
        catch
        {
            // ignore buffer errors
        }
    }

    private void OnStopped(object? sender, StoppedEventArgs e)
    {
        lock (_sync)
        {
            _capture?.Dispose();
            _capture = null;
            _buffered = null;
            _pipeline = null;
        }
    }

    private async Task PumpAsync(CancellationToken ct)
    {
        var buffer = new float[FrameSizeSamples];

        while (!ct.IsCancellationRequested)
        {
            ISampleProvider? pipeline;
            Stopwatch? sw;
            lock (_sync)
            {
                pipeline = _pipeline;
                sw = _sw;
            }

            if (pipeline == null || sw == null)
            {
                await Task.Delay(50, ct);
                continue;
            }

            int read = 0;
            try
            {
                read = pipeline.Read(buffer, 0, buffer.Length);
            }
            catch
            {
                await Task.Delay(20, ct);
                continue;
            }

            if (read < buffer.Length)
            {
                await Task.Delay(10, ct);
                continue;
            }

            var samples = new float[buffer.Length];
            Array.Copy(buffer, samples, buffer.Length);
            FrameArrived?.Invoke(this, new AudioFrameEventArgs(samples, TargetSampleRateHz, sw.Elapsed));
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            _cts?.Cancel();
        }

        try { _pumpTask?.Wait(TimeSpan.FromSeconds(1)); } catch { /* ignore */ }

        lock (_sync)
        {
            _cts?.Dispose();
            _cts = null;

            if (_capture != null)
            {
                try { _capture.StopRecording(); } catch { /* ignore */ }
                _capture.DataAvailable -= OnDataAvailable;
                _capture.RecordingStopped -= OnStopped;
                _capture.Dispose();
                _capture = null;
            }
        }
    }
}

