using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SpeechProcessing;

namespace E2ETests;

[TestFixture]
public sealed class WhisperSmokeTests
{
    private static string FindRepoRoot(string startDir)
    {
        DirectoryInfo? cur = new(startDir);
        while (cur != null)
        {
            if (Directory.Exists(Path.Combine(cur.FullName, "App", "models", "whisper")))
            {
                return cur.FullName;
            }
            cur = cur.Parent;
        }

        throw new DirectoryNotFoundException("Repo root not found (expected 'App/models/whisper' folder).");
    }

    [Test]
    [Description("Validates Whisper can transcribe some non-empty text from the offline WAV fixture.")]
#pragma warning disable CA1707 // Compatibility with existing CI filter naming
    public async Task Test_Whisper_Transcribes_NonEmpty_FromFixture()
#pragma warning restore CA1707
    {
        string repoRoot = FindRepoRoot(TestContext.CurrentContext.TestDirectory);
        string modelPath = Path.Combine(repoRoot, "App", "models", "whisper", "ggml-tiny.bin");
        Assert.That(File.Exists(modelPath), Is.True, $"Missing Whisper model at {modelPath}");

        string wav = Path.Combine(TestContext.CurrentContext.TestDirectory, "audio", "marshall_plan_speech.wav");
        Assert.That(File.Exists(wav), Is.True, $"Missing WAV fixture at {wav}");

        using WaveFileReader reader = new(wav);
        ISampleProvider sample = reader.ToSampleProvider();
        if (sample.WaveFormat.Channels == 2)
        {
            sample = new StereoToMonoSampleProvider(sample) { LeftVolume = 0.5f, RightVolume = 0.5f };
        }

        const int targetSr = 16000;
        if (sample.WaveFormat.SampleRate != targetSr)
        {
            sample = new WdlResamplingSampleProvider(sample, targetSr);
        }

        // Take up to ~8 seconds.
        int n = targetSr * 8;
        float[] buf = new float[n];
        int read = sample.Read(buf, 0, buf.Length);
        float[] pcm = read == buf.Length ? buf : [.. buf.Take(read)];

        using WhisperTranscriber t = new(modelPath);
        using CancellationTokenSource cts = new(TimeSpan.FromSeconds(60));
        string text = await t.TranscribeAsync(pcm, "en", cts.Token);

        TestContext.Out.WriteLine($"Whisper text len={text?.Length ?? 0}: {text}");
        Assert.That(text, Is.Not.Null.And.Not.Empty);
    }
}
