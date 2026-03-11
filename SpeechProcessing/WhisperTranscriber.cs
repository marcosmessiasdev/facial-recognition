using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace SpeechProcessing;

public sealed class WhisperTranscriber : IDisposable
{
    private readonly Whisper.net.WhisperFactory _factory;

    public WhisperTranscriber(string modelPath)
    {
        _factory = Whisper.net.WhisperFactory.FromPath(modelPath);
    }

    public void Dispose() => _factory.Dispose();

    public async Task<string> TranscribeAsync(float[] pcm16kMono, string? language, CancellationToken ct)
    {
        if (pcm16kMono.Length == 0) return "";

        await using var processor = _factory
            .CreateBuilder()
            .WithLanguage(string.IsNullOrWhiteSpace(language) ? "auto" : language)
            .Build();

        var parts = new List<string>();

        await foreach (var segment in processor.ProcessAsync(pcm16kMono, ct))
        {
            if (!string.IsNullOrWhiteSpace(segment.Text))
                parts.Add(segment.Text.Trim());
        }

        return string.Join(" ", parts);
    }
}

