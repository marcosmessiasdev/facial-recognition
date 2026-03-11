using System;
using System.Threading;
using System.Threading.Tasks;

namespace SpeechProcessing;

public sealed class StreamingWhisperPipeline : IDisposable
{
    private readonly VadSegmenter _segmenter;
    private readonly WhisperTranscriber _transcriber;
    private readonly string _language;
    private readonly SemaphoreSlim _oneAtATime = new(1, 1);

    public event EventHandler<TranscriptSegment>? TranscriptReady;

    public StreamingWhisperPipeline(string ggmlModelPath, string language, int hangoverMs, int maxSegmentSeconds)
    {
        _language = string.IsNullOrWhiteSpace(language) ? "auto" : language.Trim();
        _transcriber = new WhisperTranscriber(ggmlModelPath);
        _segmenter = new VadSegmenter(sampleRateHz: 16000, hangoverMs: hangoverMs, maxSegmentSeconds: maxSegmentSeconds);
        _segmenter.SegmentReady += OnSegmentReady;
        _segmenter.Start();
    }

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
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                var text = await _transcriber.TranscribeAsync(seg.Samples, _language, cts.Token);
                if (!string.IsNullOrWhiteSpace(text))
                    TranscriptReady?.Invoke(this, new TranscriptSegment(seg.Start, seg.End, text));
            }
            catch
            {
                // best-effort transcription; ignore failures
            }
            finally
            {
                _oneAtATime.Release();
            }
        });
    }

    public void Dispose()
    {
        _segmenter.SegmentReady -= OnSegmentReady;
        _segmenter.Dispose();
        _transcriber.Dispose();
        _oneAtATime.Dispose();
    }
}

