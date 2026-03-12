using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using WindowCapture;
using FramePipeline;
using FaceTracking;
using AudioProcessing;
using SpeakerDiarization;
using MeetingAnalytics;
using SpeechProcessing;
using IdentityStore;
using OverlayRenderer;
using FacialRecognition.Domain;
using OpenCvSharp;
using System.Windows;
using Config;
using Logging;
using VisionEngine.Stages;
using VisionEngine.Services;
using Microsoft.Extensions.DependencyInjection;

namespace VisionEngine;

/// <summary>
/// The central orchestrator of the facial recognition system, managing the multi-stage vision pipeline.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Coordinates frame capture, face detection, multi-person tracking, and asynchronous AI analysis 
/// (recognition, emotion, gender, age) to provide real-time facial insights.
///
/// Responsibilities:
/// - Manage the lifecycle of all AI models and capture services.
/// - Orchestrate the flow of VisionFrames from capture to processing to UI overlay.
/// - Implement load balancing and throttling logic for heavy AI inference tasks.
/// - Synchronize the visual overlay with the tracked target window.
/// - Manage the identity database connection for person matching.
///
/// Dependencies:
/// - GraphicsCaptureService (Frame source)
/// - FaceAiSharp/SCRFD (Detection)
/// - ArcFace/OnnxRuntime (Recognition)
/// - IdentityStore (Persistence)
/// - OverlayRenderer (UI Output)
///
/// Architectural Role:
/// Manager / Orchestrator. It acts as the "brain" of the application, connecting all functional modules.
///
/// Constraints:
/// - Frame processing is handled on a background task to keep the capture and UI threads responsive.
/// - Drops frames if the processing queue is saturated to ensure low latency.
/// </remarks>
public partial class VisionPipeline : IDisposable
{
    private bool _disposed;
    private CancellationTokenSource? _cancellationTokenSource;
    private Task? _pipelineTask;
    private IntPtr _targetHwnd;

    // Config — loaded from appsettings.json
    private readonly AppConfig _cfg;

    // Modules
    private readonly GraphicsCaptureService _captureService;
    private IAudioCapture? _audioCapture;
    private SileroVad? _vad;
    private VadStateMachine? _vadStateMachine;
    private bool _vadSpeechActive;
    private float _vadSpeechProb;
    private DateTime _lastVadLogUtc = DateTime.MinValue;
    private DateTime _lastAudioStatsLogUtc = DateTime.MinValue;
    private DateTime? _audioBaseUtc;
    private TimeSpan _lastAudioOffset;
    private readonly MeetingAnalyticsEngine _analytics;
    private readonly IFrameStagePipelineBuilder _pipelineBuilder;
    private StreamingWhisperPipeline? _stt;
    private volatile string? _lastTranscript;
    private OnlineSpeakerDiarizer? _diarizer;
    private int? _activeAudioSpeakerId;
    private float _activeAudioSpeakerConfidence;
    private volatile bool _acceptAudioFrames;
    private BoundingBoxOverlay? _overlay;
    private readonly bool _overlayDebugLabels;

    private readonly object _audioChunksSync = new();
    private readonly List<(TimeSpan Offset, float[] Samples)> _audioChunks = new();

    private IReadOnlyList<IFrameStage> _frameStages = [];

    // Frame Queue -> Producer/Consumer (bounded to drop old frames and keep latency low)
    private readonly BlockingCollection<VisionFrame> _frameQueue;

    /// <summary>
    /// Initializes a new instance of the VisionPipeline class with a specific configuration.
    /// </summary>
    /// <param name="cfg">The configuration settings for the vision system.</param>
    public VisionPipeline(AppConfig cfg, IServiceProvider serviceProvider, MeetingAnalyticsEngine analytics)
    {
        _cfg = cfg;
        _overlayDebugLabels = string.Equals(_cfg.OverlayLabelMode, "debug", StringComparison.OrdinalIgnoreCase);
        _pipelineBuilder = serviceProvider.GetRequiredService<IFrameStagePipelineBuilder>();
        _analytics = analytics;
        _frameQueue = new BlockingCollection<VisionFrame>(boundedCapacity: 2);
        _captureService = new GraphicsCaptureService();
        _captureService.RawFrameArrived += OnRawFrameArrived;

        AppLogger.Instance.Information("VisionPipeline created. Config: FPS={Fps}, RecognitionInterval={Ri}",
            _cfg.CaptureFps, _cfg.RecognitionIntervalFrames);

        try
        {
            string dbPath = Environment.GetEnvironmentVariable("identity_db") ??
                            Environment.GetEnvironmentVariable("IDENTITY_DB_PATH") ??
                            "identity.db";
            AppLogger.Instance.Information("IdentityStore configured. Db={DbPath}", dbPath);
        }
        catch
        {
            // ignore best-effort diagnostics
        }
    }

    /// <summary>
    /// Loads all required AI models from the filesystem and prepares the processing modules.
    /// </summary>
    /// <exception cref="FileNotFoundException">Thrown when a critical model file (e.g., SCRFD) is missing.</exception>
    public void Initialize()
    {
        string basedir = AppDomain.CurrentDomain.BaseDirectory;
        _frameStages = _pipelineBuilder.Build();

        if (_cfg.EnableTranscription)
        {
            string whisperPath = Path.Combine(basedir, _cfg.ModelWhisperGgml);
            if (File.Exists(whisperPath))
            {
                _stt = new StreamingWhisperPipeline(
                    whisperPath,
                    _cfg.TranscriptionLanguage,
                    _cfg.TranscriptionHangoverMs,
                    _cfg.TranscriptionMaxSegmentSeconds,
                    _cfg.TranscriptionMinSegmentMs);
                _stt.TranscriptReady += OnTranscriptReady;
                _stt.TranscriptionFailed += OnTranscriptionFailed;
                _stt.SpeechSegmentReady += OnSpeechSegmentReady;
                AppLogger.Instance.Information("Whisper transcription enabled: {Path}", whisperPath);
            }
            else
            {
                AppLogger.Instance.Warning("Whisper model not found at {Path} — transcription disabled", whisperPath);
            }
        }

        if (_cfg.EnableSpeakerDiarization)
        {
            string embPath = Path.Combine(basedir, _cfg.ModelSpeakerEmbedding);
            if (File.Exists(embPath))
            {
                _diarizer = new OnlineSpeakerDiarizer(
                    embPath,
                    sampleRateHz: _cfg.AudioVadSampleRateHz,
                    windowMs: _cfg.DiarizationWindowMs,
                    hopMs: _cfg.DiarizationHopMs,
                    assignThreshold: _cfg.DiarizationAssignThreshold,
                    hangoverMs: _cfg.TranscriptionHangoverMs);
                _diarizer.SegmentReady += OnAudioSpeakerSegmentReady;
                _diarizer.ActiveSpeakerUpdated += OnActiveAudioSpeakerUpdated;
                AppLogger.Instance.Information("Speaker diarization enabled: {Path}", embPath);
            }
            else
            {
                AppLogger.Instance.Warning("Speaker embedding model not found at {Path} — diarization disabled", embPath);
            }
        }

        if (_cfg.EnableAudioVad)
        {
            string vadPath = Path.Combine(basedir, _cfg.ModelSileroVad);
            if (File.Exists(vadPath))
            {
                _vad = new SileroVad(vadPath);
                AppLogger.Instance.Information("SileroVad IO: {Info}", _vad.DebugInfo);
                _vadStateMachine = new VadStateMachine(
                    minSpeechMs: _cfg.AudioVadMinSpeechMs,
                    minSilenceMs: _cfg.AudioVadMinSilenceMs,
                    hangoverMs: _cfg.AudioVadHangoverMs);

                string source = (_cfg.AudioSource ?? "loopback").Trim();

                if (string.Equals(source, "test", StringComparison.OrdinalIgnoreCase))
                {
                    string? file = _cfg.AudioTestFile;
                    if (string.IsNullOrWhiteSpace(file))
                    {
                        throw new InvalidOperationException("AudioSource is 'test' but AudioTestFile is not set.");
                    }

                    string resolved = Path.IsPathRooted(file) ? file : Path.Combine(basedir, file);
                    _audioCapture = new TestAudioCapture(
                        resolved,
                        loop: _cfg.AudioTestLoop,
                        initialSilenceMs: _cfg.AudioTestInitialSilenceMs,
                        targetSampleRateHz: _cfg.AudioVadSampleRateHz,
                        frameSizeSamples: _cfg.AudioVadFrameSizeSamples);
                }
                else
                {
                    _audioCapture = string.Equals(source, "microphone", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(source, "mic", StringComparison.OrdinalIgnoreCase)
                        ? new MicrophoneAudioCapture(_cfg.AudioVadSampleRateHz, _cfg.AudioVadFrameSizeSamples, _cfg.AudioMicrophoneDevice)
                        : new LoopbackAudioCapture(_cfg.AudioVadSampleRateHz, _cfg.AudioVadFrameSizeSamples, _cfg.AudioLoopbackDevice);
                }

                _audioCapture.FrameArrived += OnAudioFrameArrived;
                AppLogger.Instance.Information("SileroVad loaded: {Path}", vadPath);
            }
            else
            {
                AppLogger.Instance.Warning("Silero VAD model not found at {Path} — audio VAD disabled", vadPath);
            }
        }

        // Frame stages are composed via DI (see IFrameStagePipelineBuilder).
    }

    /// <summary>
    /// The main background processing loop that consumes frames and executes the AI pipeline.
    /// </summary>
    private async Task ProcessFramesAsync(CancellationToken ct)
    {
        long frameCount = 0;

        while (!ct.IsCancellationRequested)
        {
            try
            {
                if (!_frameQueue.TryTake(out VisionFrame? frame, 100, ct))
                {
                    continue;
                }

                using (frame)
                {
                    if (frame.Mat == null || _frameStages.Count == 0)
                    {
                        continue;
                    }

                    frameCount++;
                    if (frameCount % 100 == 0)
                    {
                        AppLogger.Instance.Debug("Pipeline alive — {Count} frames processed", frameCount);
                    }

                    FrameContext ctxFrame = new()
                    {
                        Frame = frame,
                        FrameCount = frameCount,
                        NowUtc = DateTime.UtcNow,
                        VadSpeechActive = _vadSpeechActive,
                        VadSpeechProb = _vadSpeechProb,
                        LastAudioOffset = _lastAudioOffset,
                        ActiveAudioSpeakerId = _activeAudioSpeakerId,
                        ActiveAudioSpeakerConfidence = _activeAudioSpeakerConfidence,
                        TryGetAudioWindow = (endOffset, window) =>
                        {
                            bool ok = TryGetRecentAudioWindow(endOffset, window, out float[] audio);
                            return (ok, audio);
                        }
                    };

                    foreach (IFrameStage stage in _frameStages)
                    {
                        stage.Process(ctxFrame);
                    }

                    List<Track> tracks = ctxFrame.Tracks;

                    // 4. Align overlay with the target window using GetWindowRect
                    _ = GetWindowRect(_targetHwnd, out RECT winRect);
                    int wLeft = winRect.Left;
                    int wTop = winRect.Top;
                    int wWidth = winRect.Right - winRect.Left;
                    int wHeight = winRect.Bottom - winRect.Top;

                    // 5. Push to UI thread — never block here
                    List<Track> snapshot = [.. tracks];
                    string? hudText = null;
                    try
                    {
                        string line1 = $"Audio: {_cfg.AudioSource} | VAD={_vadSpeechActive} prob={_vadSpeechProb:0.000}";
                        int? audioSpk = _activeAudioSpeakerId;
                        if (audioSpk.HasValue)
                        {
                            line1 += $" | DiarSpk={audioSpk.Value} conf={_activeAudioSpeakerConfidence:0.00}";
                        }

                        IReadOnlyList<(string SpeakerKey, string? DisplayName, double Seconds)>? top = _analytics?.GetSpeakingTimeSoFar(DateTime.UtcNow, top: 3);
                        if (top != null && top.Count > 0)
                        {
                            string topStr = string.Join(" | ", top.Select(t =>
                                $"{(string.IsNullOrWhiteSpace(t.DisplayName) ? t.SpeakerKey : t.DisplayName)} {TimeSpan.FromSeconds(t.Seconds):mm\\:ss}"));
                            line1 += $" | Top: {topStr}";
                        }

                        string? line2 = null;
                        IReadOnlyList<(string SpeakerKey, string? DisplayName, double Score)>? part = _analytics?.GetParticipationSoFar(DateTime.UtcNow, top: 3);
                        if (part != null && part.Count > 0)
                        {
                            string partStr = string.Join(" | ", part.Select(p =>
                                $"{(string.IsNullOrWhiteSpace(p.DisplayName) ? p.SpeakerKey : p.DisplayName)} {p.Score:0}"));
                            line2 = $"Participation: {partStr}";
                        }

                        string? line3 = _lastTranscript;
                        if (string.IsNullOrWhiteSpace(line2) && string.IsNullOrWhiteSpace(line3))
                        {
                            hudText = line1;
                        }
                        else
                        {
                            hudText = !string.IsNullOrWhiteSpace(line2) && string.IsNullOrWhiteSpace(line3)
                                ? line1 + Environment.NewLine + line2
                                : string.IsNullOrWhiteSpace(line2) && !string.IsNullOrWhiteSpace(line3)
                                ? line1 + Environment.NewLine + "STT: " + line3
                                : line1 + Environment.NewLine + line2 + Environment.NewLine + "STT: " + line3;
                        }
                    }
                    catch { /* ignore */ }
                    _ = Application.Current.Dispatcher.BeginInvoke(() =>
                    {
                        _overlay?.UpdateTracks(snapshot, wWidth, wHeight, wLeft, wTop, _overlayDebugLabels, hudText);
                    });
                }
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                AppLogger.Instance.Error(ex, "Unhandled error in frame processing loop");
            }
        }

        AppLogger.Instance.Information("Frame processing loop stopped after {Count} frames", frameCount);
    }

    [LibraryImport("user32.dll", EntryPoint = "GetWindowRect", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [StructLayout(LayoutKind.Sequential)]
    private struct RECT { public int Left, Top, Right, Bottom; }
}
