using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
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
using MeetingAnalytics;
using FaceLandmarks;
using SpeechProcessing;
using IdentityStore;
using OverlayRenderer;
using OcvRect = OpenCvSharp.Rect;
using OpenCvSharp;
using System.Windows;
using Config;
using Logging;

namespace VisionEngine
{
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
    public class VisionPipeline : IDisposable
    {
        private CancellationTokenSource? _cancellationTokenSource;
        private Task? _pipelineTask;
        private IntPtr _targetHwnd;

        // Config — loaded from appsettings.json
        private readonly AppConfig _cfg;

        // Modules
        private readonly GraphicsCaptureService _captureService;
        private FaceDetector?       _faceDetector;
        private ArcFaceRecognizer?  _recognizer;
        private EmotionClassifier?  _emotionClassifier;
        private AgeClassifier?      _ageClassifier;
        private GenderClassifier?   _genderClassifier;
        private GenderAgeClassifier? _genderAgeClassifier;
        private IAudioCapture? _audioCapture;
        private SileroVad? _vad;
        private bool _vadSpeechActive;
        private float _vadSpeechProb;
        private DateTime _lastVadLogUtc = DateTime.MinValue;
        private DateTime? _audioBaseUtc;

        private MouthMotionAnalyzer? _mouthAnalyzer;
        private ActiveSpeakerDetector? _activeSpeaker;
        private MeetingAnalyticsEngine? _analytics;
        private FaceMeshLandmarker? _faceMesh;
        private StreamingWhisperPipeline? _stt;
        private volatile string? _lastTranscript;
        private DateTime _lastAnalyticsUpdateUtc = DateTime.MinValue;
        private readonly FaceTracker      _tracker;
        private readonly PersonRepository _personRepo = new();
        private BoundingBoxOverlay? _overlay;

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
            _cfg  = cfg;
            _tracker     = new FaceTracker(_cfg.IouThreshold, _cfg.MaxMissedFrames);
            _frameQueue  = new BlockingCollection<VisionFrame>(boundedCapacity: 2);
            _captureService = new GraphicsCaptureService();
            _captureService.RawFrameArrived += OnRawFrameArrived;

            AppLogger.Instance.Information("VisionPipeline created. Config: FPS={Fps}, RecognitionInterval={Ri}",
                _cfg.CaptureFps, _cfg.RecognitionIntervalFrames);
        }

        /// <summary>
        /// Loads all required AI models from the filesystem and prepares the processing modules.
        /// </summary>
        /// <exception cref="System.IO.FileNotFoundException">Thrown when a critical model file (e.g., SCRFD) is missing.</exception>
        public void Initialize()
        {
            var basedir = System.AppDomain.CurrentDomain.BaseDirectory;

            string scrfdPath    = System.IO.Path.Combine(basedir, _cfg.ModelScrfd);
            string arcfacePath  = System.IO.Path.Combine(basedir, _cfg.ModelArcface);
            string fer2013Path  = System.IO.Path.Combine(basedir, _cfg.ModelFer2013);
            string genderAgePath = System.IO.Path.Combine(basedir, _cfg.ModelGenderAge);
            string genderPath   = System.IO.Path.Combine(basedir, _cfg.ModelGender);
            string agePath      = System.IO.Path.Combine(basedir, _cfg.ModelAge);

            if (!System.IO.File.Exists(scrfdPath))
                throw new System.IO.FileNotFoundException($"SCRFD model not found: {scrfdPath}");

            _faceDetector = new FaceDetector(scrfdPath);
            AppLogger.Instance.Information("FaceDetector loaded: {Path}", scrfdPath);

            if (System.IO.File.Exists(arcfacePath))
            {
                _recognizer = new ArcFaceRecognizer(arcfacePath);
                AppLogger.Instance.Information("ArcFaceRecognizer loaded: {Path}", arcfacePath);
            }
            else
                AppLogger.Instance.Warning("ArcFace model not found at {Path} — recognition disabled", arcfacePath);

            if (System.IO.File.Exists(fer2013Path))
            {
                _emotionClassifier = new EmotionClassifier(fer2013Path);
                AppLogger.Instance.Information("EmotionClassifier loaded: {Path}", fer2013Path);
            }
            else
                AppLogger.Instance.Warning("Emotion model not found at {Path} — emotion analysis disabled", fer2013Path);

            bool wantAttributes = _cfg.EnableGenderPrediction || _cfg.EnableAgePrediction;
            if (wantAttributes && System.IO.File.Exists(genderAgePath))
            {
                _genderAgeClassifier = new GenderAgeClassifier(genderAgePath);
                AppLogger.Instance.Information("GenderAgeClassifier loaded: {Path}", genderAgePath);
            }
            else
            {
                if (wantAttributes)
                    AppLogger.Instance.Warning("GenderAge model not found at {Path} — falling back to separate gender/age models", genderAgePath);

                if (_cfg.EnableGenderPrediction)
                {
                    if (System.IO.File.Exists(genderPath))
                    {
                        _genderClassifier = new GenderClassifier(genderPath);
                        AppLogger.Instance.Information("GenderClassifier loaded: {Path}", genderPath);
                    }
                    else
                        AppLogger.Instance.Warning("Gender model not found at {Path} — gender prediction disabled", genderPath);
                }

                if (_cfg.EnableAgePrediction)
                {
                    if (System.IO.File.Exists(agePath))
                    {
                        _ageClassifier = new AgeClassifier(agePath);
                        AppLogger.Instance.Information("AgeClassifier loaded: {Path}", agePath);
                    }
                    else
                        AppLogger.Instance.Warning("Age model not found at {Path} — age prediction disabled", agePath);
                }
            }

            if (_cfg.EnableFaceMeshLandmarks)
            {
                var faceMeshPath = System.IO.Path.Combine(basedir, _cfg.ModelFaceMesh);
                if (System.IO.File.Exists(faceMeshPath))
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
                var whisperPath = System.IO.Path.Combine(basedir, _cfg.ModelWhisperGgml);
                if (System.IO.File.Exists(whisperPath))
                {
                    _stt = new StreamingWhisperPipeline(
                        whisperPath,
                        _cfg.TranscriptionLanguage,
                        _cfg.TranscriptionHangoverMs,
                        _cfg.TranscriptionMaxSegmentSeconds);
                    _stt.TranscriptReady += OnTranscriptReady;
                    AppLogger.Instance.Information("Whisper transcription enabled: {Path}", whisperPath);
                }
                else
                {
                    AppLogger.Instance.Warning("Whisper model not found at {Path} — transcription disabled", whisperPath);
                }
            }

            if (_cfg.EnableAudioVad)
            {
                var vadPath = System.IO.Path.Combine(basedir, _cfg.ModelSileroVad);
                if (System.IO.File.Exists(vadPath))
                {
                    _vad = new SileroVad(vadPath);

                    var source = (_cfg.AudioSource ?? "loopback").Trim();
                    if (string.Equals(source, "microphone", StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(source, "mic", StringComparison.OrdinalIgnoreCase))
                    {
                        _audioCapture = new MicrophoneAudioCapture(_cfg.AudioVadSampleRateHz, _cfg.AudioVadFrameSizeSamples, _cfg.AudioMicrophoneDevice);
                    }
                    else
                    {
                        _audioCapture = new LoopbackAudioCapture(_cfg.AudioVadSampleRateHz, _cfg.AudioVadFrameSizeSamples, _cfg.AudioLoopbackDevice);
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
        /// Starts the capture and processing pipeline for a specific target window.
        /// </summary>
        /// <param name="targetHwnd">The Window Handle (HWND) of the application to track.</param>
        public void Start(IntPtr targetHwnd)
        {
            _targetHwnd = targetHwnd;

            Application.Current.Dispatcher.Invoke(() =>
            {
                _overlay = new BoundingBoxOverlay();
                _overlay.Show();
            });

            _captureService.StartCapture(targetHwnd);
            AppLogger.Instance.Information("Capture started for HWND={Hwnd}", targetHwnd);

            try
            {
                _audioCapture?.Start();
                if (_audioCapture != null)
                    AppLogger.Instance.Information("Audio capture started (VAD enabled). Source={Source} Device={Device}",
                        _cfg.AudioSource, _audioCapture.SelectedDeviceName ?? "Default");
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Warning(ex, "Audio capture failed to start — VAD disabled for this run");
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _pipelineTask = Task.Run(
                () => ProcessFramesAsync(_cancellationTokenSource.Token),
                _cancellationTokenSource.Token);
        }

        /// <summary>
        /// Handles incoming audio frames and performs Voice Activity Detection (VAD).
        /// </summary>
        private void OnAudioFrameArrived(object? sender, AudioFrameEventArgs e)
        {
            var vad = _vad;
            if (vad == null) return;

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

            bool active = prob >= _cfg.AudioVadSpeechThreshold;
            bool stateChanged = active != _vadSpeechActive;
            _vadSpeechActive = active;
            _vadSpeechProb = prob;

            var now = DateTime.UtcNow;
            _audioBaseUtc ??= now - e.Offset;
            if (stateChanged || (now - _lastVadLogUtc) > TimeSpan.FromSeconds(2))
            {
                _lastVadLogUtc = now;
                AppLogger.Instance.Debug("VAD speech={Speech} prob={Prob:0.00}", _vadSpeechActive, prob);
            }

            try
            {
                _stt?.PushFrame(e.Samples, e.Offset, active);
            }
            catch
            {
                // ignore
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
                    var baseUtc = _audioBaseUtc ?? DateTime.UtcNow;
                    var startUtc = baseUtc + seg.Start;
                    var endUtc = baseUtc + seg.End;
                    _analytics.AddUtterance(startUtc, endUtc, seg.Text);
                }
            }
            catch
            {
                // ignore
            }
        }

        /// <summary>
        /// Handles incoming raw video frames from the capture service.
        /// </summary>
        private void OnRawFrameArrived(object? sender, (byte[] data, int width, int height, int stride) args)
        {
            // Producer — drop frame if queue is full to keep latency minimal
            if (_frameQueue.Count < _frameQueue.BoundedCapacity)
            {
                var visionFrame = new VisionFrame(args.data, args.width, args.height, args.stride);
                _frameQueue.TryAdd(visionFrame);
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
                    if (!_frameQueue.TryTake(out var frame, 100, ct)) continue;

                    using (frame)
                    {
                        if (_faceDetector == null || frame.Mat == null) continue;

                        frameCount++;
                        if (frameCount % 100 == 0)
                            AppLogger.Instance.Debug("Pipeline alive — {Count} frames processed", frameCount);

                        // 1. Detect bounding boxes (SCRFD)
                        var boxes = _faceDetector.Detect(frame);

                        // Log face count every frame for E2E test log validation
                        if (boxes.Count > 0 || frameCount % 30 == 0)
                            AppLogger.Instance.Debug("Faces detected: {Count} | Frame: {Frame}", boxes.Count, frameCount);

                        // 2. Update tracker — get stable, ID-assigned face list
                        var tracks = _tracker.Update(boxes);

                        // 2a. Mouth motion (cheap visual proxy for speaking)
                        if (_mouthAnalyzer != null)
                        {
                            var nowUtc = DateTime.UtcNow;
                            foreach (var track in tracks)
                            {
                                track.FramesSinceLandmarks++;

                                var rectForMotion = new OcvRect(
                                    Math.Max(0, track.Box.X),
                                    Math.Max(0, track.Box.Y),
                                    Math.Min(track.Box.Width, frame.Mat.Width - track.Box.X),
                                    Math.Min(track.Box.Height, frame.Mat.Height - track.Box.Y));

                                if (rectForMotion.Width < 24 || rectForMotion.Height < 24)
                                {
                                    track.MouthMotionScore = 0f;
                                    continue;
                                }

                                using var faceCropForMotion = new Mat(frame.Mat, rectForMotion);

                                OpenCvSharp.Rect? mouthRoi = null;
                                float? openRatio = track.MouthOpenRatio > 0 ? track.MouthOpenRatio : null;

                                if (_faceMesh != null && track.FramesSinceLandmarks >= _cfg.FaceMeshIntervalFrames)
                                {
                                    try
                                    {
                                        if (_faceMesh.TryGetMouthMetrics(faceCropForMotion, out var ratio, out var roi))
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

                                var lms = track.Box.Landmarks;
                                if (mouthRoi == null && lms != null && lms.Length >= 5)
                                {
                                    // SCRFD landmarks typically: [leftEye, rightEye, nose, leftMouth, rightMouth]
                                    var leftMouth = lms[3];
                                    var rightMouth = lms[4];

                                    float lmX1 = leftMouth.X - track.Box.X;
                                    float lmY1 = leftMouth.Y - track.Box.Y;
                                    float lmX2 = rightMouth.X - track.Box.X;
                                    float lmY2 = rightMouth.Y - track.Box.Y;

                                    float cx = (lmX1 + lmX2) / 2f;
                                    float cy = (lmY1 + lmY2) / 2f;
                                    float mouthWidth = MathF.Max(8f, MathF.Sqrt((lmX2 - lmX1) * (lmX2 - lmX1) + (lmY2 - lmY1) * (lmY2 - lmY1)));

                                    int rw = (int)(mouthWidth * 2.0f);
                                    int rh = (int)(mouthWidth * 1.1f);
                                    int rx = (int)(cx - rw / 2f);
                                    int ry = (int)(cy - rh * 0.35f);

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
                        foreach (var track in tracks)
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
                                continue;

                            var rect = new OcvRect(
                                Math.Max(0, track.Box.X),
                                Math.Max(0, track.Box.Y),
                                Math.Min(track.Box.Width, frame.Mat.Width - track.Box.X),
                                Math.Min(track.Box.Height, frame.Mat.Height - track.Box.Y));

                            if (rect.Width < 16 || rect.Height < 16) continue;

                            using var faceCrop = new Mat(frame.Mat, rect);

                            // 3a. ArcFace recognition + SQLite lookup
                            if (doRecognition && _recognizer != null)
                            {
                                track.FramesSinceRecognition = 0;

                                try
                                {
                                    var embedding = _recognizer.GetEmbedding(faceCrop);
                                    var (person, sim) = _personRepo.FindBestMatch(
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
                                    var (emotion, conf) = _emotionClassifier.Classify(faceCrop);
                                    track.EmotionLabel = $"{emotion} {conf:P0}";
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
                                    var result = _genderAgeClassifier.Predict(frame.Mat, track.Box);
                                    if (result.HasValue)
                                    {
                                        var (gender, genderConf, ageLabel) = result.Value;

                                        if (_cfg.EnableGenderPrediction)
                                            track.GenderLabel = $"{gender} {genderConf:P0}";

                                        if (_cfg.EnableAgePrediction)
                                            track.AgeLabel = $"Age {ageLabel}";
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
                                    var (gender, conf) = _genderClassifier.Classify(faceCrop);
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
                                    var (_, label, conf) = _ageClassifier.Classify(faceCrop);
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
                        foreach (var t in tracks)
                            t.IsSpeaking = activeSpeakerId.HasValue && t.Id == activeSpeakerId.Value;

                        // 3f. Meeting analytics (timeline + metrics)
                        if (_analytics != null)
                        {
                            var nowUtc = DateTime.UtcNow;
                            // Avoid excessive segment churn from frame-level noise.
                            if ((nowUtc - _lastAnalyticsUpdateUtc) > TimeSpan.FromMilliseconds(100))
                            {
                                _lastAnalyticsUpdateUtc = nowUtc;
                                _analytics.Update(tracks, activeSpeakerId, nowUtc);
                            }
                        }

                        // 4. Align overlay with the target window using GetWindowRect
                        GetWindowRect(_targetHwnd, out var winRect);
                        int wLeft   = winRect.Left;
                        int wTop    = winRect.Top;
                        int wWidth  = winRect.Right  - winRect.Left;
                        int wHeight = winRect.Bottom - winRect.Top;

                        // 5. Push to UI thread — never block here
                        var snapshot = new System.Collections.Generic.List<Track>(tracks);
                        string? hudText = null;
                        try
                        {
                            var line1 = $"Audio: {_cfg.AudioSource} | VAD={_vadSpeechActive} prob={_vadSpeechProb:0.00}";
                            var top = _analytics?.GetSpeakingTimeSoFar(DateTime.UtcNow, top: 3);
                            if (top != null && top.Count > 0)
                            {
                                var topStr = string.Join(" | ", top.Select(t =>
                                    $"{(string.IsNullOrWhiteSpace(t.DisplayName) ? t.SpeakerKey : t.DisplayName)} {TimeSpan.FromSeconds(t.Seconds):mm\\:ss}"));
                                line1 += $" | Top: {topStr}";
                            }

                            var line2 = _lastTranscript;
                            hudText = string.IsNullOrWhiteSpace(line2) ? line1 : line1 + Environment.NewLine + "STT: " + line2;
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

        /// <summary>
        /// Stops all capture and processing tasks and closes the UI overlay.
        /// </summary>
        public void Stop()
        {
            AppLogger.Instance.Information("Stopping VisionPipeline...");
            _cancellationTokenSource?.Cancel();
            _pipelineTask?.Wait(TimeSpan.FromSeconds(3));
            _captureService.StopCapture();

            try { _audioCapture?.Stop(); } catch { /* ignore */ }

            try
            {
                if (_analytics != null)
                {
                    var basedir = System.AppDomain.CurrentDomain.BaseDirectory;
                    var session = _analytics.StopAndBuild(DateTime.UtcNow);
                    var path = _analytics.Persist(session, basedir);
                    AppLogger.Instance.Information("Meeting session saved: {Path}", path);
                }
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Warning(ex, "Failed to persist meeting analytics session");
            }

            Application.Current.Dispatcher.Invoke(() =>
            {
                _overlay?.Close();
                _overlay = null;
            });

            while (_frameQueue.TryTake(out var f)) f.Dispose();
            AppLogger.Instance.Information("VisionPipeline stopped.");
        }

        /// <summary>
        /// Releases all resources used by the VisionPipeline, including models and capture services.
        /// </summary>
        public void Dispose()
        {
            Stop();
            _captureService.Dispose();
            if (_audioCapture != null) _audioCapture.FrameArrived -= OnAudioFrameArrived;
            _audioCapture?.Dispose();
            _vad?.Dispose();
            _mouthAnalyzer?.Dispose();
            _faceMesh?.Dispose();
            if (_stt != null) _stt.TranscriptReady -= OnTranscriptReady;
            _stt?.Dispose();
            _analytics = null;
            _faceDetector?.Dispose();
            _recognizer?.Dispose();
            _emotionClassifier?.Dispose();
            _genderAgeClassifier?.Dispose();
            _ageClassifier?.Dispose();
            _genderClassifier?.Dispose();
            _personRepo.Dispose();
            _frameQueue.Dispose();
        }

        [DllImport("user32.dll")]
        private static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);

        private struct RECT { public int Left, Top, Right, Bottom; }
    }
}
