using Whisper.net;

namespace SpeechProcessing;

public sealed class WhisperTranscriber(string modelPath) : IDisposable
{
    private readonly WhisperFactory _factory = WhisperFactory.FromPath(modelPath);

    public void Dispose()
    {
        _factory.Dispose();
        GC.SuppressFinalize(this);
    }

    public async Task<string> TranscribeAsync(float[] pcm16kMono, string? language, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(pcm16kMono);

        if (pcm16kMono.Length == 0)
        {
            return "";
        }

        await using WhisperProcessor processor = _factory
            .CreateBuilder()
            .WithLanguage(string.IsNullOrWhiteSpace(language) ? "auto" : language)
            .Build();

        List<string> parts = new();

        await foreach (SegmentData segment in processor.ProcessAsync(pcm16kMono, ct))
        {
            if (!string.IsNullOrWhiteSpace(segment.Text))
            {
                parts.Add(segment.Text.Trim());
            }
        }

        return string.Join(" ", parts);
    }
}
