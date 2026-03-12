using System.Windows;
using FramePipeline;
using Logging;
using MeetingAnalytics;
using OpenCvSharp;
using OverlayRenderer;

namespace VisionEngine;

public partial class VisionPipeline
{
    /// <summary>
    /// Starts the capture and processing pipeline for a specific target window.
    /// </summary>
    /// <param name="targetHwnd">The Window Handle (HWND) of the application to track.</param>
    public void Start(IntPtr targetHwnd)
    {
        _targetHwnd = targetHwnd;
        _acceptAudioFrames = true;

        Application.Current.Dispatcher.Invoke(() =>
        {
            _overlay = new BoundingBoxOverlay();
            _overlay.Show();
        });

        _captureService.StartCapture(targetHwnd);
        AppLogger.Instance.Information("Capture started for HWND={Hwnd}", targetHwnd);

        try
        {
            _vad?.ResetState();
            _vadStateMachine?.Reset();
            _audioCapture?.Start();
            if (_audioCapture != null)
            {
                AppLogger.Instance.Information("Audio capture started (VAD enabled). Source={Source} Device={Device}",
                    _cfg.AudioSource, _audioCapture.SelectedDeviceName ?? "Default");
            }
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
    /// Stops all capture and processing tasks and closes the UI overlay.
    /// </summary>
    public void Stop()
    {
        AppLogger.Instance.Information("Stopping VisionPipeline...");
        _acceptAudioFrames = false;
        _cancellationTokenSource?.Cancel();
        _ = (_pipelineTask?.Wait(TimeSpan.FromSeconds(3)));

        try
        {
            // Capture shutdown can occasionally block due to GraphicsCapture/COM timing.
            // Do not let it prevent meeting analytics persistence or clean stop logs.
            Task stopCap = Task.Run(() => _captureService.StopCapture());
            if (!stopCap.Wait(TimeSpan.FromSeconds(2)))
            {
                AppLogger.Instance.Warning("Capture stop timed out; continuing shutdown (resources will be released on process exit).");
            }
        }
        catch (Exception ex)
        {
            AppLogger.Instance.Warning(ex, "Capture stop failed; continuing shutdown");
        }

        try
        {
            if (_audioCapture != null)
            {
                _audioCapture.FrameArrived -= OnAudioFrameArrived;
            }
        }
        catch { /* ignore */ }

        try { _audioCapture?.StopCapture(); } catch { /* ignore */ }

        try
        {
            // Best-effort flush of pending audio-derived events before persisting the session JSON.
            try { _diarizer?.Flush(_lastAudioOffset); } catch { /* ignore */ }
            try { _stt?.StopAndDrain(TimeSpan.FromSeconds(25)); } catch { /* ignore */ }

            if (_analytics != null)
            {
                string basedir = AppDomain.CurrentDomain.BaseDirectory;
                MeetingSession session = _analytics.StopAndBuild(DateTime.UtcNow);
                string path = MeetingAnalyticsEngine.Persist(session, basedir);
                AppLogger.Instance.Information("Meeting session saved: {Path}", path);

                try
                {
                    string reportPath = MeetingReportGenerator.PersistHtml(session, basedir);
                    string transcriptPath = MeetingReportGenerator.PersistTranscript(session, basedir);
                    string timelinePath = MeetingReportGenerator.PersistSpeakerTimelineCsv(session, basedir);
                    AppLogger.Instance.Information("Meeting report saved: {Path}", reportPath);
                    AppLogger.Instance.Information("Meeting transcript saved: {Path}", transcriptPath);
                    AppLogger.Instance.Information("Meeting timeline saved: {Path}", timelinePath);
                }
                catch (Exception ex)
                {
                    AppLogger.Instance.Warning(ex, "Failed to persist meeting report artifacts");
                }
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

        while (_frameQueue.TryTake(out VisionFrame? f))
        {
            f.Dispose();
        }

        AppLogger.Instance.Information("VisionPipeline stopped.");
    }

    /// <summary>
    /// Releases all resources used by the VisionPipeline, including models and capture services.
    /// </summary>
    public void Dispose()
    {
        Stop();
        _captureService.Dispose();
        if (_audioCapture != null)
        {
            _audioCapture.FrameArrived -= OnAudioFrameArrived;
        }

        _audioCapture?.Dispose();
        _vad?.Dispose();
        _mouthAnalyzer?.Dispose();
        _faceMesh?.Dispose();
        if (_stt != null)
        {
            _stt.TranscriptReady -= OnTranscriptReady;
            _stt.TranscriptionFailed -= OnTranscriptionFailed;
            _stt.SpeechSegmentReady -= OnSpeechSegmentReady;
        }

        _stt?.Dispose();
        if (_diarizer != null)
        {
            _diarizer.SegmentReady -= OnAudioSpeakerSegmentReady;
            _diarizer.ActiveSpeakerUpdated -= OnActiveAudioSpeakerUpdated;
        }

        _diarizer?.Dispose();
        _talkNet?.Dispose();
        foreach (Queue<Mat> q in _talkNetFramesByTrack.Values)
        {
            while (q.Count > 0)
            {
                q.Dequeue().Dispose();
            }
        }
        _talkNetFramesByTrack.Clear();
        _analytics = null;
        _faceDetector?.Dispose();
        _recognizer?.Dispose();
        _emotionClassifier?.Dispose();
        _genderAgeClassifier?.Dispose();
        _ageClassifier?.Dispose();
        _genderClassifier?.Dispose();
        _personRepo.Dispose();
        _frameQueue.Dispose();

        GC.SuppressFinalize(this);
    }
}

