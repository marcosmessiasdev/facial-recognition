using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using NAudio.CoreAudioApi;

namespace AudioProcessing;

/// <summary>
/// Captures system audio loopback and provides a stream of processed mono audio frames.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Enables the application to "hear" the system audio output for tasks like Voice Activity Detection (VAD).
///
/// Responsibilities:
/// - Initialize and manage WASAPI loopback capture.
/// - Convert multi-channel system audio to mono format.
/// - Resample audio to a target frequency (e.g., 16kHz) required by AI models.
/// - Implement an asynchronous pump that emits fixed-size audio frames via events.
///
/// Dependencies:
/// - NAudio (Audio capture and processing library)
///
/// Architectural Role:
/// Infrastructure Component / Audio Source.
///
/// Constraints:
/// - Only captures system audio; does not capture from physical microphones unless configured otherwise.
/// - Requires valid audio output device to be active on the system.
/// </remarks>
public sealed class LoopbackAudioCapture : IAudioCapture
{
    private readonly object _sync = new();
    private readonly string? _deviceSelector;

    private WasapiLoopbackCapture? _capture;
    private BufferedWaveProvider? _buffered;
    private ISampleProvider? _pipeline;
    private CancellationTokenSource? _cts;
    private Task? _pumpTask;
    private Stopwatch? _sw;

    /// <summary>
    /// Occurs when a new processed audio frame is available.
    /// </summary>
    public event EventHandler<AudioFrameEventArgs>? FrameArrived;

    /// <summary>
    /// Gets the target sample rate in Hertz.
    /// </summary>
    public int TargetSampleRateHz { get; }

    /// <summary>
    /// Gets the target number of samples per frame.
    /// </summary>
    public int FrameSizeSamples { get; }

    public string? SelectedDeviceName { get; private set; }

    /// <summary>
    /// Initializes a new instance of the LoopbackAudioCapture class.
    /// </summary>
    /// <param name="targetSampleRateHz">The sample rate to convert the captured audio to.</param>
    /// <param name="frameSizeSamples">The size of each audio frame to emit.</param>
    public LoopbackAudioCapture(int targetSampleRateHz = 16000, int frameSizeSamples = 512, string? deviceSelector = null)
    {
        if (targetSampleRateHz <= 0) throw new ArgumentOutOfRangeException(nameof(targetSampleRateHz));
        if (frameSizeSamples <= 0) throw new ArgumentOutOfRangeException(nameof(frameSizeSamples));
        TargetSampleRateHz = targetSampleRateHz;
        FrameSizeSamples = frameSizeSamples;
        _deviceSelector = string.IsNullOrWhiteSpace(deviceSelector) ? null : deviceSelector.Trim();
    }

    /// <summary>
    /// Starts the system audio loopback capture session.
    /// </summary>
    public void Start()
    {
        lock (_sync)
        {
            if (_capture != null) return;

            var device = ResolveDevice(_deviceSelector);
            SelectedDeviceName = device?.FriendlyName;
            _capture = device != null ? new WasapiLoopbackCapture(device) : new WasapiLoopbackCapture();
            _buffered = new BufferedWaveProvider(_capture.WaveFormat)
            {
                DiscardOnBufferOverflow = true
            };

            _capture.DataAvailable += OnDataAvailable;
            _capture.RecordingStopped += OnStopped;

            var sample = _buffered.ToSampleProvider();

            if (sample.WaveFormat.Channels == 2)
            {
                var mono = new StereoToMonoSampleProvider(sample)
                {
                    LeftVolume = 0.5f,
                    RightVolume = 0.5f
                };
                sample = mono;
            }
            else if (sample.WaveFormat.Channels > 2)
            {
                // Best-effort: take first channel only.
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
            var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);

            if (!string.IsNullOrWhiteSpace(selector))
            {
                var byId = devices.FirstOrDefault(d => string.Equals(d.ID, selector, StringComparison.OrdinalIgnoreCase));
                if (byId != null) return byId;

                var byName = devices.FirstOrDefault(d => d.FriendlyName.Contains(selector, StringComparison.OrdinalIgnoreCase));
                if (byName != null) return byName;
            }

            return enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Stops the system audio loopback capture session.
    /// </summary>
    public void Stop()
    {
        lock (_sync)
        {
            if (_capture == null) return;
            try { _capture.StopRecording(); } catch { /* ignore */ }
        }
    }

    /// <summary>
    /// Handles raw audio data buffers from the WASAPI capture session.
    /// </summary>
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
            // ignore buffer errors; capture should keep running
        }
    }

    /// <summary>
    /// Handles cleanup when the recording session stops.
    /// </summary>
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

    /// <summary>
    /// Background task that pulls data through the audio pipeline and emits frames.
    /// </summary>
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

    /// <summary>
    /// Releases all audio resources and stops background tasks.
    /// </summary>
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
