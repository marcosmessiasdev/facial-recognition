namespace SpeechProcessing;

/// <summary>
/// Connects a continuous audio stream into asynchronous Voice Activity segmented transcription blocks.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Acts as the bridging logic between real-time, low-latency audio capture and the 
/// high-latency, batch-oriented Whisper.net STT engine.
///
/// Responsibilities:
/// - Buffer audio using a VadSegmenter.
/// - Convert completed VAD segments into asynchronous transcription tasks.
/// - Serialize transcription requests using a SemaphoreSlim to prevent CPU stalling on overlapping speech.
/// - Handle GGML model initialization.
///
/// Dependencies:
/// - WhisperTranscriber (Whisper.net wrapper)
/// - VadSegmenter (Audio buffering)
///
/// Architectural Role:
/// AI Service Component / Streaming facade.
/// </remarks>
public sealed class StreamingWhisperPipeline : IDisposable
{
    private readonly VadSegmenter _segmenter;
    private readonly WhisperTranscriber _transcriber;
    private readonly string _language;
    private readonly SemaphoreSlim _oneAtATime = new(1, 1);

    /// <summary>
    /// Event fired when a chunk of audio has finished converting to a text sequence.
    /// </summary>
    public event EventHandler<TranscriptSegment>? TranscriptReady;

    /// <summary>
    /// Instantiates the streaming Whisper pipeline.
    /// </summary>
    /// <param name="ggmlModelPath">Path to local Whisper GGML inference model.</param>
    /// <param name="language">Language code or "auto".</param>
    /// <param name="hangoverMs">Ms of silence allowed before finishing a sentence.</param>
    /// <param name="maxSegmentSeconds">Absolute maximum duration of an audio chunk string before splitting.</param>
    public StreamingWhisperPipeline(string ggmlModelPath, string language, int hangoverMs, int maxSegmentSeconds)
    {
        _language = string.IsNullOrWhiteSpace(language) ? "auto" : language.Trim();
        _transcriber = new WhisperTranscriber(ggmlModelPath);
        _segmenter = new VadSegmenter(sampleRateHz: 16000, hangoverMs: hangoverMs, maxSegmentSeconds: maxSegmentSeconds);
        _segmenter.SegmentReady += OnSegmentReady;
        _segmenter.Start();
    }

    /// <summary>
    /// Queues a chunk of frames to the segmenter context.
    /// </summary>
    /// <param name="samples16kMono">Raw float samples.</param>
    /// <param name="timestamp">Starting timestamp of the array.</param>
    /// <param name="speechActive">VAD gate signal.</param>
    public void PushFrame(float[] samples16kMono, TimeSpan timestamp, bool speechActive)
    {
        _segmenter.Push(samples16kMono, timestamp, speechActive);
    }

    private void OnSegmentReady(object? sender, (TimeSpan Start, TimeSpan End, float[] Samples) seg)
    {
        _ = Task.Run(async () =>
        {
            await _oneAtATime.WaitAsync();
            try
            {
                using CancellationTokenSource cts = new(TimeSpan.FromSeconds(30));
                string text = await _transcriber.TranscribeAsync(seg.Samples, _language, cts.Token);
                if (!string.IsNullOrWhiteSpace(text))
                {
                    TranscriptReady?.Invoke(this, new TranscriptSegment(seg.Start, seg.End, text));
                }
            }
            catch
            {
                // best-effort transcription; ignore failures
            }
            finally
            {
                _ = _oneAtATime.Release();
            }
        });
    }

    /// <summary>
    /// Flushes tasks and disposes native speech and semaphore instances.
    /// </summary>
    public void Dispose()
    {
        _segmenter.SegmentReady -= OnSegmentReady;
        _segmenter.Dispose();
        _transcriber.Dispose();
        _oneAtATime.Dispose();
    }
}

