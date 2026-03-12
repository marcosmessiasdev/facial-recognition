using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using WindowCapture;
using FramePipeline;
using FaceDetection;
using FaceTracking;
using FaceRecognition;
using EmotionAnalysis;
using AgeAnalysis;
using GenderAnalysis;
using FaceAttributes;
using AudioProcessing;
using SpeakerDetection;
using SpeakerDiarization;
using MeetingAnalytics;
using FaceLandmarks;
using SpeechProcessing;
using IdentityStore;
using OverlayRenderer;
using FacialRecognition.Domain;
using OcvRect = OpenCvSharp.Rect;
using OpenCvSharp;
using System.Windows;
using Config;
using Logging;

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
    private CancellationTokenSource? _cancellationTokenSource;
    private Task? _pipelineTask;
    private IntPtr _targetHwnd;

    // Config — loaded from appsettings.json
    private readonly AppConfig _cfg;

    // Modules
    private readonly GraphicsCaptureService _captureService;
    private FaceDetector? _faceDetector;
    private ArcFaceRecognizer? _recognizer;
    private EmotionClassifier? _emotionClassifier;
    private AgeClassifier? _ageClassifier;
    private GenderClassifier? _genderClassifier;
    private GenderAgeClassifier? _genderAgeClassifier;
    private IAudioCapture? _audioCapture;
    private SileroVad? _vad;
    private VadStateMachine? _vadStateMachine;
    private bool _vadSpeechActive;
    private float _vadSpeechProb;
    private DateTime _lastVadLogUtc = DateTime.MinValue;
    private DateTime _lastAudioStatsLogUtc = DateTime.MinValue;
    private DateTime? _audioBaseUtc;
    private TimeSpan _lastAudioOffset;

    private MouthMotionAnalyzer? _mouthAnalyzer;
    private ActiveSpeakerDetector? _activeSpeaker;
    private MeetingAnalyticsEngine? _analytics;
    private FaceMeshLandmarker? _faceMesh;
    private StreamingWhisperPipeline? _stt;
    private volatile string? _lastTranscript;
    private OnlineSpeakerDiarizer? _diarizer;
    private int? _activeAudioSpeakerId;
    private float _activeAudioSpeakerConfidence;
    private DateTime _lastAnalyticsUpdateUtc = DateTime.MinValue;
    private volatile bool _acceptAudioFrames;
    private readonly FaceTracker _tracker;
    private readonly PersonRepository _personRepo = new();
    private BoundingBoxOverlay? _overlay;
    private readonly bool _overlayDebugLabels;

    private TalkNetAsdModel? _talkNet;
    private readonly object _audioChunksSync = new();
    private readonly List<(TimeSpan Offset, float[] Samples)> _audioChunks = new();
    private readonly Dictionary<int, Queue<Mat>> _talkNetFramesByTrack = new();

    // Frame Queue -> Producer/Consumer (bounded to drop old frames and keep latency low)
    private readonly BlockingCollection<VisionFrame> _frameQueue;

    /// <summary>
    /// Initializes a new instance of the VisionPipeline class using default configurations.
    /// </summary>
    public VisionPipeline() : this(AppConfig.Load()) { }

    /// <summary>
    /// Initializes a new instance of the VisionPipeline class with a specific configuration.
    /// </summary>
    /// <param name="cfg">The configuration settings for the vision system.</param>
    public VisionPipeline(AppConfig cfg)
    {
        _cfg = cfg;
        _overlayDebugLabels = string.Equals(_cfg.OverlayLabelMode, "debug", StringComparison.OrdinalIgnoreCase);
        _tracker = new FaceTracker(_cfg.IouThreshold, _cfg.MaxMissedFrames);
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
            int count = _personRepo.GetAll().Count;
            AppLogger.Instance.Information("IdentityStore ready. Db={DbPath} Persons={Count}", dbPath, count);
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

        string scrfdPath = Path.Combine(basedir, _cfg.ModelScrfd);
        string arcfacePath = Path.Combine(basedir, _cfg.ModelArcface);
        string fer2013Path = Path.Combine(basedir, _cfg.ModelFer2013);
        string genderAgePath = Path.Combine(basedir, _cfg.ModelGenderAge);
        string genderPath = Path.Combine(basedir, _cfg.ModelGender);
        string agePath = Path.Combine(basedir, _cfg.ModelAge);

        if (!File.Exists(scrfdPath))
        {
            throw new FileNotFoundException($"SCRFD model not found: {scrfdPath}");
        }

        _faceDetector = new FaceDetector(scrfdPath);
        AppLogger.Instance.Information("FaceDetector loaded: {Path}", scrfdPath);

        if (File.Exists(arcfacePath))
        {
            _recognizer = new ArcFaceRecognizer(arcfacePath);
            AppLogger.Instance.Information("ArcFaceRecognizer loaded: {Path}", arcfacePath);
        }
        else
        {
            AppLogger.Instance.Warning("ArcFace model not found at {Path} — recognition disabled", arcfacePath);
        }

        if (File.Exists(fer2013Path))
        {
            _emotionClassifier = new EmotionClassifier(fer2013Path);
            AppLogger.Instance.Information("EmotionClassifier loaded: {Path}", fer2013Path);
        }
        else
        {
            AppLogger.Instance.Warning("Emotion model not found at {Path} — emotion analysis disabled", fer2013Path);
        }

        bool wantAttributes = _cfg.EnableGenderPrediction || _cfg.EnableAgePrediction;
        if (wantAttributes && File.Exists(genderAgePath))
        {
            _genderAgeClassifier = new GenderAgeClassifier(genderAgePath);
            AppLogger.Instance.Information("GenderAgeClassifier loaded: {Path}", genderAgePath);
        }
        else
        {
            if (wantAttributes)
            {
                AppLogger.Instance.Warning("GenderAge model not found at {Path} — falling back to separate gender/age models", genderAgePath);
            }

            if (_cfg.EnableGenderPrediction)
            {
                if (File.Exists(genderPath))
                {
                    _genderClassifier = new GenderClassifier(genderPath);
                    AppLogger.Instance.Information("GenderClassifier loaded: {Path}", genderPath);
                }
                else
                {
                    AppLogger.Instance.Warning("Gender model not found at {Path} — gender prediction disabled", genderPath);
                }
            }

            if (_cfg.EnableAgePrediction)
            {
                if (File.Exists(agePath))
                {
                    _ageClassifier = new AgeClassifier(agePath);
                    AppLogger.Instance.Information("AgeClassifier loaded: {Path}", agePath);
                }
                else
                {
                    AppLogger.Instance.Warning("Age model not found at {Path} — age prediction disabled", agePath);
                }
            }
        }

        if (_cfg.EnableFaceMeshLandmarks)
        {
            string faceMeshPath = Path.Combine(basedir, _cfg.ModelFaceMesh);
            if (File.Exists(faceMeshPath))
            {
                _faceMesh = new FaceMeshLandmarker(faceMeshPath);
                AppLogger.Instance.Information("FaceMeshLandmarker loaded: {Path}", faceMeshPath);
            }
            else
            {
                AppLogger.Instance.Warning("FaceMesh model not found at {Path} — using heuristic mouth motion only", faceMeshPath);
            }
        }

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

        if (_cfg.EnableTalkNetAsd)
        {
            string talknetPath = Path.Combine(basedir, _cfg.ModelTalkNetAsd);
            if (File.Exists(talknetPath))
            {
                _talkNet = new TalkNetAsdModel(talknetPath);
                AppLogger.Instance.Information("TalkNet ASD enabled: {Path}", talknetPath);
            }
            else
            {
                AppLogger.Instance.Warning("TalkNet ASD model not found at {Path} — falling back to heuristics", talknetPath);
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
                _mouthAnalyzer = new MouthMotionAnalyzer();
                _activeSpeaker = new ActiveSpeakerDetector();
                _analytics = new MeetingAnalyticsEngine();
            }
            else
            {
                AppLogger.Instance.Warning("Silero VAD model not found at {Path} — audio VAD disabled", vadPath);
            }
        }
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
                    if (_faceDetector == null || frame.Mat == null)
                    {
                        continue;
                    }

                    frameCount++;
                    if (frameCount % 100 == 0)
                    {
                        AppLogger.Instance.Debug("Pipeline alive — {Count} frames processed", frameCount);
                    }

                    // 1. Detect bounding boxes (SCRFD)
                    List<BoundingBox> boxes = _faceDetector.Detect(frame);

                    // Log face count every frame for E2E test log validation
                    if (boxes.Count > 0 || frameCount % 30 == 0)
                    {
                        AppLogger.Instance.Debug("Faces detected: {Count} | Frame: {Frame}", boxes.Count, frameCount);
                    }

                    // 2. Update tracker — get stable, ID-assigned face list
                    List<Track> tracks = _tracker.Update(boxes);

                    // 2a. TalkNet ASD buffer + inference (multimodal)
                    UpdateTalkNetAsd(frame, tracks);

                    // 2b. Mouth motion (cheap visual proxy for speaking)
                    UpdateMouthMotion(frame, tracks);

                    // 3. Run classifiers + active speaker + analytics
                    int? activeSpeakerId = RunVisionInferenceAndAnalytics(frame, tracks);

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

    private static OcvRect ClampRect(OcvRect r, int w, int h)
    {
        int x = Math.Clamp(r.X, 0, Math.Max(0, w - 1));
        int y = Math.Clamp(r.Y, 0, Math.Max(0, h - 1));
        int rw = Math.Clamp(r.Width, 1, w - x);
        int rh = Math.Clamp(r.Height, 1, h - y);
        return new OcvRect(x, y, rw, rh);
    }

    private static OcvRect ExpandRect(OcvRect r, float marginRatio, int w, int h)
    {
        if (marginRatio <= 0f)
        {
            return ClampRect(r, w, h);
        }

        int mx = (int)MathF.Round(r.Width * marginRatio);
        int my = (int)MathF.Round(r.Height * marginRatio);
        OcvRect expanded = new(r.X - mx, r.Y - my, r.Width + (2 * mx), r.Height + (2 * my));
        return ClampRect(expanded, w, h);
    }

    private static (Emotion e1, float p1, Emotion e2, float p2) Top2Emotion(float[] probs)
    {
        if (probs.Length == 0)
        {
            return (Emotion.Neutral, 0f, Emotion.Neutral, 0f);
        }

        int n = Math.Min(probs.Length, 8);
        int best1 = 0;
        int best2 = 0;
        float pBest1 = probs[0];
        float pBest2 = float.NegativeInfinity;

        for (int i = 1; i < n; i++)
        {
            float p = probs[i];
            if (p > pBest1)
            {
                best2 = best1;
                pBest2 = pBest1;
                best1 = i;
                pBest1 = p;
            }
            else if (p > pBest2)
            {
                best2 = i;
                pBest2 = p;
            }
        }

        Emotion e1 = (Emotion)best1;
        Emotion e2 = (Emotion)best2;
        float p1 = float.IsFinite(pBest1) ? pBest1 : 0f;
        float p2 = float.IsFinite(pBest2) ? pBest2 : 0f;
        return (e1, p1, e2, p2);
    }

    private static string FormatEmotionProbs(float[] probs)
    {
        int n = Math.Min(probs.Length, 8);
        if (n <= 0)
        {
            return string.Empty;
        }

        string[] parts = new string[n];
        for (int i = 0; i < n; i++)
        {
            parts[i] = $"{(Emotion)i}={probs[i]:0.000}";
        }

        return string.Join(" ", parts);
    }

    private void UpdateTalkNetAsd(VisionFrame frame, List<Track> tracks)
    {
        TalkNetAsdModel? talkNet = _talkNet;
        if (talkNet == null || !_cfg.EnableTalkNetAsd || frame.Mat == null || frame.Mat.Empty())
        {
            foreach (Track t in tracks)
            {
                t.TalkNetSpeakingProb = 0f;
                t.FramesSinceAsd++;
            }

            // Still prune to active tracks to avoid unbounded growth.
            PruneTalkNetQueues(tracks);
            return;
        }

        DateTime nowUtc = DateTime.UtcNow;
        int windowFrames = Math.Clamp(_cfg.TalkNetWindowFrames, 5, 60);
        float fps = Math.Clamp(_cfg.CaptureFps, 10, 60);
        TimeSpan window = TimeSpan.FromSeconds(windowFrames / fps);

        // 1) Append latest visual frame per track into temporal queues.
        foreach (Track t in tracks)
        {
            t.FramesSinceAsd++;

            if (t.Box.Width <= 2 || t.Box.Height <= 2)
            {
                continue;
            }

            OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
            rect = ClampRect(rect, frame.Mat.Width, frame.Mat.Height);
            if (rect.Width < 8 || rect.Height < 8)
            {
                continue;
            }

            using Mat face = new(frame.Mat, rect);
            using Mat gray = new();
            if (face.Channels() == 4)
            {
                Cv2.CvtColor(face, gray, ColorConversionCodes.BGRA2GRAY);
            }
            else if (face.Channels() == 3)
            {
                Cv2.CvtColor(face, gray, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                face.CopyTo(gray);
            }

            Mat resized = new();
            Cv2.Resize(gray, resized, new OpenCvSharp.Size(112, 112));

            if (!_talkNetFramesByTrack.TryGetValue(t.Id, out Queue<Mat>? q))
            {
                q = new Queue<Mat>(windowFrames + 2);
                _talkNetFramesByTrack[t.Id] = q;
            }

            q.Enqueue(resized);
            while (q.Count > windowFrames)
            {
                Mat old = q.Dequeue();
                old.Dispose();
            }
        }

        PruneTalkNetQueues(tracks);

        // 2) Run inference on tracks that have a full window and a recent audio window.
        // Use the most recent audio offset as end bound.
        if (_cfg.EnableAudioVad && _audioCapture != null && TryGetRecentAudioWindow(_lastAudioOffset, window, out float[] audio))
        {
            foreach (Track t in tracks)
            {
                if (!_talkNetFramesByTrack.TryGetValue(t.Id, out Queue<Mat>? q) || q.Count < windowFrames)
                {
                    t.TalkNetSpeakingProb = 0f;
                    continue;
                }

                // Throttle TalkNet per track for performance.
                if (t.FramesSinceAsd < 2)
                {
                    continue;
                }

                float[] probs;
                try
                {
                    probs = talkNet.Predict(audio, q.ToArray(), fps);
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "TalkNet ASD inference failed (best-effort)");
                    t.TalkNetSpeakingProb = 0f;
                    continue;
                }

                float v = probs.Length > 0 ? probs[^1] : 0f;
                float p = (v < 0f || v > 1f) ? Sigmoid(v) : v;
                p = Math.Clamp(p, 0f, 1f);
                t.TalkNetSpeakingProb = p;
                t.FramesSinceAsd = 0;
            }
        }
        else
        {
            foreach (Track t in tracks)
            {
                t.TalkNetSpeakingProb = 0f;
            }
        }

        // Best-effort telemetry.
        if ((nowUtc - _lastVadLogUtc) > TimeSpan.FromSeconds(5))
        {
            _lastVadLogUtc = nowUtc;
            float max = tracks.Count > 0 ? tracks.Max(t => t.TalkNetSpeakingProb) : 0f;
            if (max > 0.0001f)
            {
                AppLogger.Instance.Debug("TalkNet ASD maxProb={Prob:0.000}", max);
            }
        }
    }

    private void UpdateMouthMotion(VisionFrame frame, List<Track> tracks)
    {
        MouthMotionAnalyzer? mouth = _mouthAnalyzer;
        if (mouth == null || frame.Mat == null || frame.Mat.Empty())
        {
            foreach (Track t in tracks)
            {
                t.MouthMotionScore = 0f;
                t.MouthOpenRatio = 0f;
                t.FramesSinceLandmarks++;
            }

            return;
        }

        DateTime nowUtc = DateTime.UtcNow;
        FaceMeshLandmarker? faceMesh = _faceMesh;
        bool useFaceMesh = _cfg.EnableFaceMeshLandmarks && faceMesh != null;

        foreach (Track t in tracks)
        {
            t.FramesSinceLandmarks++;

            OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
            rect = ClampRect(rect, frame.Mat.Width, frame.Mat.Height);
            if (rect.Width < 16 || rect.Height < 16)
            {
                t.MouthMotionScore = 0f;
                continue;
            }

            using Mat face = new(frame.Mat, rect);
            using Mat faceClone = face.Clone();

            OpenCvSharp.Rect? mouthRoi = null;
            float? openRatio = null;

            if (useFaceMesh && t.FramesSinceLandmarks >= _cfg.FaceMeshIntervalFrames)
            {
                try
                {
                    if (faceMesh!.TryGetMouthMetrics(faceClone, out float r, out OpenCvSharp.Rect roi))
                    {
                        mouthRoi = roi;
                        openRatio = r;
                        t.MouthOpenRatio = r;
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "FaceMesh mouth metrics failed (best-effort)");
                }

                t.FramesSinceLandmarks = 0;
            }

            float score;
            try
            {
                score = mouth.Update(t.Id, faceClone, mouthRoi, openRatio, nowUtc);
            }
            catch
            {
                score = 0f;
            }

            t.MouthMotionScore = score;
        }

        mouth.PruneToActiveTracks(tracks.Select(t => t.Id));
    }

    private int? RunVisionInferenceAndAnalytics(VisionFrame frame, List<Track> tracks)
    {
        DateTime nowUtc = DateTime.UtcNow;

        foreach (Track t in tracks)
        {
            t.FramesSinceRecognition++;
            t.FramesSinceEmotion++;
            t.FramesSinceEmotionDebugLog++;
            t.FramesSinceGender++;
            t.FramesSinceAge++;
        }

        // Recognition (ArcFace + IdentityStore)
        if (_recognizer != null && tracks.Count > 0)
        {
            foreach (Track t in tracks)
            {
                if (t.FramesSinceRecognition < _cfg.RecognitionIntervalFrames)
                {
                    continue;
                }

                try
                {
                    OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                    rect = ClampRect(rect, frame.Mat.Width, frame.Mat.Height);
                    if (rect.Width < 16 || rect.Height < 16)
                    {
                        continue;
                    }

                    using Mat face = new(frame.Mat, rect);
                    using Mat crop = face.Clone();
                    float[] emb = _recognizer.GetEmbedding(crop);

                    (Person? person, float sim) = _personRepo.FindBestMatch(emb, threshold: (float)_cfg.RecognitionThreshold);
                    if (person != null && !string.IsNullOrWhiteSpace(person.Name))
                    {
                        string newName = $"{person.Name} ({sim:P0})";
                        bool changed = string.IsNullOrWhiteSpace(t.PersonName) ||
                                       !t.PersonName.StartsWith(person.Name, StringComparison.OrdinalIgnoreCase);
                        t.PersonName = newName;
                        if (changed)
                        {
                            AppLogger.Instance.Information("Recognized: {Name} sim={Sim:0.00} track={TrackId}", person.Name, sim, t.Id);
                        }
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Warning(ex, "Recognition inference error");
                }
                finally
                {
                    t.FramesSinceRecognition = 0;
                }
            }
        }

        // Emotion (FER2013)
        if (_emotionClassifier != null)
        {
            foreach (Track t in tracks)
            {
                if (t.FramesSinceEmotion < _cfg.EmotionInterval)
                {
                    continue;
                }

                try
                {
                    OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                    rect = ExpandRect(rect, _cfg.EmotionCropMarginRatio, frame.Mat.Width, frame.Mat.Height);
                    if (rect.Width < 16 || rect.Height < 16)
                    {
                        continue;
                    }

                    using Mat face = new(frame.Mat, rect);
                    using Mat crop = face.Clone();

                    float[] probs = _emotionClassifier.GetProbabilities(crop);
                    (Emotion e1, float p1, Emotion e2, float p2) = Top2Emotion(probs);

                    t.EmotionLabel = $"{e1} {p1:P0}  P2 {e2} {p2:P0}";
                    _analytics?.AddEmotionSample(t, e1.ToString(), p1, nowUtc);

                    if (_cfg.EmotionDebugLogProbs &&
                        t.FramesSinceEmotionDebugLog >= Math.Max(1, _cfg.EmotionDebugLogEveryNFrames))
                    {
                        t.FramesSinceEmotionDebugLog = 0;
                        string vec = FormatEmotionProbs(probs);
                        AppLogger.Instance.Debug("Emotion probs track={TrackId}: {Vec}", t.Id, vec);
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "Emotion inference failed (best-effort)");
                }
                finally
                {
                    t.FramesSinceEmotion = 0;
                }
            }
        }

        // Attributes (gender/age)
        if (_genderAgeClassifier != null)
        {
            foreach (Track t in tracks)
            {
                if (t.FramesSinceGender < _cfg.AttributesInterval)
                {
                    continue;
                }

                try
                {
                    (string genderLabel, float genderConfidence, string ageLabel)? res = _genderAgeClassifier.Predict(frame.Mat, t.Box);
                    if (res.HasValue)
                    {
                        t.GenderLabel = $"{res.Value.genderLabel} {res.Value.genderConfidence:P0}";
                        t.AgeLabel = $"Age {res.Value.ageLabel}";
                    }
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Debug(ex, "GenderAge inference failed (best-effort)");
                }
                finally
                {
                    t.FramesSinceGender = 0;
                    t.FramesSinceAge = 0;
                }
            }
        }
        else
        {
            if (_genderClassifier != null)
            {
                foreach (Track t in tracks)
                {
                    if (t.FramesSinceGender < _cfg.GenderInterval)
                    {
                        continue;
                    }

                    try
                    {
                        OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                        rect = ClampRect(rect, frame.Mat.Width, frame.Mat.Height);
                        if (rect.Width < 16 || rect.Height < 16)
                        {
                            continue;
                        }

                        using Mat face = new(frame.Mat, rect);
                        using Mat crop = face.Clone();
                        (GenderAppearance g, float conf) = _genderClassifier.Classify(crop);
                        t.GenderLabel = $"{g} {conf:P0}";
                    }
                    catch (Exception ex)
                    {
                        AppLogger.Instance.Debug(ex, "Gender inference failed (best-effort)");
                    }
                    finally
                    {
                        t.FramesSinceGender = 0;
                    }
                }
            }

            if (_ageClassifier != null)
            {
                foreach (Track t in tracks)
                {
                    if (t.FramesSinceAge < _cfg.AgeInterval)
                    {
                        continue;
                    }

                    try
                    {
                        OcvRect rect = new(t.Box.X, t.Box.Y, t.Box.Width, t.Box.Height);
                        rect = ClampRect(rect, frame.Mat.Width, frame.Mat.Height);
                        if (rect.Width < 16 || rect.Height < 16)
                        {
                            continue;
                        }

                        using Mat face = new(frame.Mat, rect);
                        using Mat crop = face.Clone();
                        (_, string label, float conf) = _ageClassifier.Classify(crop);
                        t.AgeLabel = $"Age {label} {conf:P0}";
                    }
                    catch (Exception ex)
                    {
                        AppLogger.Instance.Debug(ex, "Age inference failed (best-effort)");
                    }
                    finally
                    {
                        t.FramesSinceAge = 0;
                    }
                }
            }
        }

        // Active speaker (TalkNet preferred, otherwise mouth motion) + analytics.
        int? activeSpeakerId = null;
        if (_activeSpeaker != null)
        {
            activeSpeakerId = _activeSpeaker.Update(tracks, _vadSpeechActive, _cfg.EnableVisualSpeakerFallback, nowUtc);
            foreach (Track t in tracks)
            {
                t.IsSpeaking = activeSpeakerId.HasValue && t.Id == activeSpeakerId.Value;
            }
        }

        try
        {
            if (_analytics != null && (nowUtc - _lastAnalyticsUpdateUtc) > TimeSpan.FromMilliseconds(120))
            {
                _lastAnalyticsUpdateUtc = nowUtc;
                _analytics.Update(tracks, activeSpeakerId, nowUtc);

                // Record audio↔face co-occurrence for mapping diarization clusters to visual tracks.
                if (_activeAudioSpeakerId.HasValue && activeSpeakerId.HasValue)
                {
                    _analytics.ObserveAudioToFaceCooccurrence(_activeAudioSpeakerId.Value, activeSpeakerId.Value);
                }
            }
        }
        catch
        {
            // ignore
        }

        return activeSpeakerId;
    }

    private void PruneTalkNetQueues(IEnumerable<Track> tracks)
    {
        HashSet<int> keep = [.. tracks.Select(t => t.Id)];
        int[] remove = [.. _talkNetFramesByTrack.Keys.Where(id => !keep.Contains(id))];
        foreach (int id in remove)
        {
            if (_talkNetFramesByTrack.TryGetValue(id, out Queue<Mat>? q))
            {
                while (q.Count > 0)
                {
                    q.Dequeue().Dispose();
                }
            }

            _ = _talkNetFramesByTrack.Remove(id);
        }
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0)
        {
            float z = MathF.Exp(-x);
            return 1f / (1f + z);
        }
        else
        {
            float z = MathF.Exp(x);
            return z / (1f + z);
        }
    }

    [LibraryImport("user32.dll", EntryPoint = "GetWindowRect", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

    [StructLayout(LayoutKind.Sequential)]
    private struct RECT { public int Left, Top, Right, Bottom; }
}
