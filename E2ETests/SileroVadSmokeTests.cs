using AudioProcessing;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace E2ETests;

[TestFixture]
public sealed class SileroVadSmokeTests
{
    private static string FindRepoRoot(string startDir)
    {
        DirectoryInfo? cur = new(startDir);
        while (cur != null)
        {
            if (Directory.Exists(Path.Combine(cur.FullName, "App", "onnx")))
            {
                return cur.FullName;
            }
            cur = cur.Parent;
        }

        throw new DirectoryNotFoundException("Repo root not found (expected 'App/onnx' folder).");
    }

    [Test]
    [Description("Validates Silero VAD produces non-trivial speech probability on the offline WAV fixture.")]
#pragma warning disable CA1707 // Compatibility with existing CI filter naming
    public void Test_SileroVad_ProducesSpeechProbability_OnFixture()
#pragma warning restore CA1707
    {
        string repoRoot = FindRepoRoot(TestContext.CurrentContext.TestDirectory);
        string modelPath = Path.Combine(repoRoot, "App", "onnx", "silero_vad.onnx");
        string wav = Path.Combine(TestContext.CurrentContext.TestDirectory, "audio", "e2e_fixture_10_words.wav");
        Assert.Multiple(() =>
        {
            Assert.That(File.Exists(modelPath), Is.True, $"Missing VAD model at {modelPath}");
            Assert.That(File.Exists(wav), Is.True, $"Missing WAV fixture at {wav}");
        });

        using SileroVad vad = new(modelPath);

        // Baseline on pure silence (state reset).
        vad.ResetState();
        float[] silence = new float[512];
        float silenceMax = 0f;
        for (int i = 0; i < 80; i++)
        {
            silenceMax = Math.Max(silenceMax, vad.GetSpeechProbability(silence, 16000));
        }

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

        const int frame = 512;
        float[] buf = new float[frame];
        vad.ResetState();
        float maxProb = 0f;
        float minProb = 1f;
        int framesRead = 0;

        // Scan the first ~6 seconds of audio.
        int maxFrames = (int)(6 * targetSr / (double)frame);
        while (framesRead < maxFrames)
        {
            int read = sample.Read(buf, 0, buf.Length);
            if (read < buf.Length)
            {
                break;
            }

            float[] chunk = new float[buf.Length];
            Array.Copy(buf, chunk, buf.Length);
            float p = vad.GetSpeechProbability(chunk, targetSr);
            maxProb = Math.Max(maxProb, p);
            minProb = Math.Min(minProb, p);
            framesRead++;
        }

        TestContext.Out.WriteLine($"Silero VAD silenceMax={silenceMax:0.000} | fixture frames={framesRead} min={minProb:0.000} max={maxProb:0.000} | {vad.DebugInfo}");

        Assert.Multiple(() =>
        {
            Assert.That(framesRead, Is.GreaterThan(10), "Fixture too short or could not be read.");
            Assert.That(silenceMax, Is.LessThan(0.05f), "VAD baseline too high on silence; model or preprocessing likely wrong.");
            Assert.That(maxProb, Is.GreaterThan(0.05f), "VAD probability never rose above 0.05 on a speech fixture.");
        });
    }
}
