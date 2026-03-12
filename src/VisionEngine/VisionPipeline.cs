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
        _tracker = new FaceTracker(_cfg.IouThreshold, _cfg.MaxMissedFrames);
        _frameQueue = new BlockingCollection<VisionFrame>(boundedCapacity: 2);
        _captureService = new GraphicsCaptureService();
        _captureService.RawFrameArrived += OnRawFrameArrived;

        AppLogger.Instance.Information("VisionPipeline created. Config: FPS={Fps}, RecognitionInterval={Ri}",
            _cfg.CaptureFps, _cfg.RecognitionIntervalFrames);
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
                    if (_talkNet != null)
                    {
                        float fps = Math.Max(1, _cfg.CaptureFps);
                        int windowFrames = Math.Clamp(_cfg.TalkNetWindowFrames, 5, 60);
                        DateTime nowUtc = DateTime.UtcNow;
                        DateTime baseUtc = _audioBaseUtc ?? nowUtc;
                        TimeSpan nowOffset = nowUtc - baseUtc;
                        TimeSpan window = TimeSpan.FromSeconds(windowFrames / fps);

                        foreach (Track track in tracks)
                        {
                            track.FramesSinceAsd++;

                            OcvRect rect = new(
                                    Math.Max(0, track.Box.X),
                                    Math.Max(0, track.Box.Y),
                                    Math.Min(track.Box.Width, frame.Mat.Width - track.Box.X),
                                    Math.Min(track.Box.Height, frame.Mat.Height - track.Box.Y));

                            if (rect.Width < 24 || rect.Height < 24)
                            {
                                continue;
                            }

                            using Mat crop = new(frame.Mat, rect);
                            using Mat gray = new();
                            if (crop.Channels() == 4)
                            {
                                Cv2.CvtColor(crop, gray, ColorConversionCodes.BGRA2GRAY);
                            }
                            else
                            {
                                Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);
                            }

                            using Mat resized = new();
                            Cv2.Resize(gray, resized, new OpenCvSharp.Size(112, 112));

                            if (!_talkNetFramesByTrack.TryGetValue(track.Id, out Queue<Mat>? q))
                            {
                                q = new Queue<Mat>(windowFrames + 2);
                                _talkNetFramesByTrack[track.Id] = q;
                            }

                            q.Enqueue(resized.Clone());
                            while (q.Count > windowFrames)
                            {
                                Mat old = q.Dequeue();
                                old.Dispose();
                            }

                            // Infer only when enough frames are buffered and at a modest cadence.
                            if (q.Count == windowFrames && track.FramesSinceAsd >= 5 && TryGetRecentAudioWindow(nowOffset, window, out float[]? audio))
                            {
                                try
                                {
                                    float[] probs = _talkNet.Predict(audio, [.. q], fps);
                                    if (probs.Length > 0)
                                    {
                                        float mean = 0f;
                                        foreach (float p in probs)
                                        {
                                            mean += p;
                                        }

                                        mean /= probs.Length;
                                        track.TalkNetSpeakingProb = mean;
                                    }
                                    track.FramesSinceAsd = 0;
                                }
                                catch (Exception ex)
                                {
                                    AppLogger.Instance.Debug(ex, "TalkNet ASD inference failed for track {Id}", track.Id);
                                    track.TalkNetSpeakingProb = 0f;
                                }
                            }
                        }

                        // Prune stale tracks.
                        HashSet<int> activeIds = [.. tracks.Select(t => t.Id)];
                        int[] keys = [.. _talkNetFramesByTrack.Keys];
                        foreach (int id in keys)
                        {
                            if (activeIds.Contains(id))
                            {
                                continue;
                            }

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

                    // 2b. Mouth motion (cheap visual proxy for speaking)
                    if (_mouthAnalyzer != null)
                    {
                        DateTime nowUtc = DateTime.UtcNow;
                        foreach (Track track in tracks)
                        {
                            track.FramesSinceLandmarks++;

                            OcvRect rectForMotion = new(
                                Math.Max(0, track.Box.X),
                                Math.Max(0, track.Box.Y),
                                Math.Min(track.Box.Width, frame.Mat.Width - track.Box.X),
                                Math.Min(track.Box.Height, frame.Mat.Height - track.Box.Y));

                            if (rectForMotion.Width < 24 || rectForMotion.Height < 24)
                            {
                                track.MouthMotionScore = 0f;
                                continue;
                            }

                            using Mat faceCropForMotion = new(frame.Mat, rectForMotion);

                            OcvRect? mouthRoi = null;
                            float? openRatio = track.MouthOpenRatio > 0 ? track.MouthOpenRatio : null;

                            if (_faceMesh != null && track.FramesSinceLandmarks >= _cfg.FaceMeshIntervalFrames)
                            {
                                try
                                {
                                    if (_faceMesh.TryGetMouthMetrics(faceCropForMotion, out float ratio, out OcvRect roi))
                                    {
                                        track.MouthOpenRatio = ratio;
                                        openRatio = ratio;
                                        mouthRoi = roi;
                                    }
                                    track.FramesSinceLandmarks = 0;
                                }
                                catch (Exception ex)
                                {
                                    AppLogger.Instance.Debug(ex, "FaceMesh landmarks failed for track {Id}", track.Id);
                                }
                            }

                            System.Drawing.PointF[]? lms = track.Box.Landmarks;
                            if (mouthRoi == null && lms != null && lms.Length >= 5)
                            {
                                // SCRFD landmarks typically: [leftEye, rightEye, nose, leftMouth, rightMouth]
                                System.Drawing.PointF leftMouth = lms[3];
                                System.Drawing.PointF rightMouth = lms[4];

                                float lmX1 = leftMouth.X - track.Box.X;
                                float lmY1 = leftMouth.Y - track.Box.Y;
                                float lmX2 = rightMouth.X - track.Box.X;
                                float lmY2 = rightMouth.Y - track.Box.Y;

                                float cx = (lmX1 + lmX2) / 2f;
                                float cy = (lmY1 + lmY2) / 2f;
                                float mouthWidth = MathF.Max(8f, MathF.Sqrt(((lmX2 - lmX1) * (lmX2 - lmX1)) + ((lmY2 - lmY1) * (lmY2 - lmY1))));

                                int rw = (int)(mouthWidth * 2.0f);
                                int rh = (int)(mouthWidth * 1.1f);
                                int rx = (int)(cx - (rw / 2f));
                                int ry = (int)(cy - (rh * 0.35f));

                                rx = Math.Clamp(rx, 0, Math.Max(0, rectForMotion.Width - 1));
                                ry = Math.Clamp(ry, 0, Math.Max(0, rectForMotion.Height - 1));
                                rw = Math.Clamp(rw, 1, rectForMotion.Width - rx);
                                rh = Math.Clamp(rh, 1, rectForMotion.Height - ry);
                                mouthRoi = new OcvRect(rx, ry, rw, rh);
                            }

                            track.MouthMotionScore = _mouthAnalyzer.Update(track.Id, faceCropForMotion, mouthRoi, openRatio, nowUtc);
                        }

                        _mouthAnalyzer.PruneToActiveTracks(tracks.Select(t => t.Id));
                    }

                    // 3. Run classifiers at configured intervals per track
                    foreach (Track track in tracks)
                    {
                        track.FramesSinceRecognition++;
                        track.FramesSinceEmotion++;
                        track.FramesSinceGender++;
                        track.FramesSinceAge++;

                        bool doRecognition = _recognizer != null &&
                                             track.FramesSinceRecognition >= _cfg.RecognitionIntervalFrames;
                        bool doEmotion = _emotionClassifier != null &&
                                         track.FramesSinceEmotion >= _cfg.EmotionInterval;

                        bool doAttributes = _genderAgeClassifier != null &&
                                            (track.FramesSinceGender >= _cfg.AttributesInterval ||
                                             track.FramesSinceAge >= _cfg.AttributesInterval);

                        bool doGender = !doAttributes &&
                                        _genderClassifier != null &&
                                        track.FramesSinceGender >= _cfg.GenderInterval;
                        bool doAge = !doAttributes &&
                                     _ageClassifier != null &&
                                     track.FramesSinceAge >= _cfg.AgeInterval;

                        if (!doRecognition && !doEmotion && !doAttributes && !doGender && !doAge)
                        {
                            continue;
                        }

                        OcvRect rect = new(
                            Math.Max(0, track.Box.X),
                            Math.Max(0, track.Box.Y),
                            Math.Min(track.Box.Width, frame.Mat.Width - track.Box.X),
                            Math.Min(track.Box.Height, frame.Mat.Height - track.Box.Y));

                        if (rect.Width < 16 || rect.Height < 16)
                        {
                            continue;
                        }

                        using Mat faceCrop = new(frame.Mat, rect);

                        // 3a. ArcFace recognition + SQLite lookup
                        if (doRecognition && _recognizer != null)
                        {
                            track.FramesSinceRecognition = 0;

                            try
                            {
                                float[] embedding = _recognizer.GetEmbedding(faceCrop);
                                (Person? person, float sim) = _personRepo.FindBestMatch(
                                    embedding, (float)_cfg.RecognitionThreshold);
                                track.PersonName = person != null
                                    ? $"{person.Name} ({sim:P0})"
                                    : "Desconhecido";
                            }
                            catch (Exception ex)
                            {
                                AppLogger.Instance.Error(ex, "Recognition inference error for track {Id}", track.Id);
                            }
                        }

                        // 3b. Emotion classification
                        if (doEmotion && _emotionClassifier != null)
                        {
                            track.FramesSinceEmotion = 0;

                            try
                            {
                                (Emotion emotion, float conf) = _emotionClassifier.Classify(faceCrop);
                                track.EmotionLabel = $"{emotion} {conf:P0}";
                                _analytics?.AddEmotionSample(track, emotion.ToString(), conf, DateTime.UtcNow);
                            }
                            catch (Exception ex)
                            {
                                AppLogger.Instance.Error(ex, "Emotion inference error for track {Id}", track.Id);
                            }
                        }

                        // 3c. Gender appearance prediction
                        if (doAttributes && _genderAgeClassifier != null)
                        {
                            track.FramesSinceGender = 0;
                            track.FramesSinceAge = 0;

                            try
                            {
                                (string genderLabel, float genderConfidence, string ageLabel)? result = _genderAgeClassifier.Predict(frame.Mat, track.Box);
                                if (result.HasValue)
                                {
                                    (string? gender, float genderConf, string? ageLabel) = result.Value;

                                    if (_cfg.EnableGenderPrediction)
                                    {
                                        track.GenderLabel = $"{gender} {genderConf:P0}";
                                    }

                                    if (_cfg.EnableAgePrediction)
                                    {
                                        track.AgeLabel = $"Age {ageLabel}";
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                AppLogger.Instance.Error(ex, "Attributes inference error for track {Id}", track.Id);
                            }
                        }

                        // 3c (fallback). Gender appearance prediction
                        if (doGender && _genderClassifier != null)
                        {
                            track.FramesSinceGender = 0;

                            try
                            {
                                (GenderAppearance gender, float conf) = _genderClassifier.Classify(faceCrop);
                                track.GenderLabel = $"{gender} {conf:P0}";
                            }
                            catch (Exception ex)
                            {
                                AppLogger.Instance.Error(ex, "Gender inference error for track {Id}", track.Id);
                            }
                        }

                        // 3d. Age bucket prediction
                        if (doAge && _ageClassifier != null)
                        {
                            track.FramesSinceAge = 0;

                            try
                            {
                                (AgeBucket _, string? label, float conf) = _ageClassifier.Classify(faceCrop);
                                track.AgeLabel = $"Age {label} {conf:P0}";
                            }
                            catch (Exception ex)
                            {
                                AppLogger.Instance.Error(ex, "Age inference error for track {Id}", track.Id);
                            }
                        }
                    }

                    // 3e. Active speaker selection (audio VAD + mouth motion)
                    int? activeSpeakerId = _activeSpeaker?.Update(tracks, _vadSpeechActive, _cfg.EnableVisualSpeakerFallback, DateTime.UtcNow);
                    foreach (Track t in tracks)
                    {
                        t.IsSpeaking = activeSpeakerId.HasValue && t.Id == activeSpeakerId.Value;
                    }

                    // 3f. Meeting analytics (timeline + metrics)
                    if (_analytics != null)
                    {
                        DateTime nowUtc = DateTime.UtcNow;
                        // Avoid excessive segment churn from frame-level noise.
                        if ((nowUtc - _lastAnalyticsUpdateUtc) > TimeSpan.FromMilliseconds(100))
                        {
                            _lastAnalyticsUpdateUtc = nowUtc;
                            _analytics.Update(tracks, activeSpeakerId, nowUtc);

                            int? audioSpk = _activeAudioSpeakerId;
                            if (audioSpk.HasValue && activeSpeakerId.HasValue)
                            {
                                _analytics.ObserveAudioToFaceCooccurrence(audioSpk.Value, activeSpeakerId.Value);
                            }
                        }
                    }

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
                        _overlay?.UpdateTracks(snapshot, wWidth, wHeight, wLeft, wTop, hudText);
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
