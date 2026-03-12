using AudioProcessing;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using SpeakerDiarization;

namespace E2ETests;

[TestFixture]
public sealed class DiarizationSmokeTests
{
    private static string FindRepoRoot(string startDir)
    {
        DirectoryInfo? cur = new(startDir);
        while (cur != null)
        {
            if (Directory.Exists(Path.Combine(cur.FullName, "src", "App", "onnx")))
            {
                return cur.FullName;
            }
            cur = cur.Parent;
        }

        throw new DirectoryNotFoundException("Repo root not found (expected 'src/App/onnx' folder).");
    }

    [Test]
    [Description("Validates diarization produces segments and at least one stable speaker ID on an offline deterministic fixture.")]
#pragma warning disable CA1707
    public void Test_Diarization_ProducesSegments_OnOfflineFixture()
#pragma warning restore CA1707
    {
        string repoRoot = FindRepoRoot(TestContext.CurrentContext.TestDirectory);
        string embeddingModelPath = Path.Combine(repoRoot, "src", "App", "onnx", "nemo_en_titanet_small.onnx");
        Assert.That(File.Exists(embeddingModelPath), Is.True, $"Missing speaker embedding model at {embeddingModelPath}");

        string wav = Path.Combine(TestContext.CurrentContext.TestDirectory, "audio", "e2e_fixture_two_speakers.wav");
        Assert.That(File.Exists(wav), Is.True, $"Missing WAV fixture at {wav}");

        // This test is intentionally independent from VAD to stay deterministic across machines.
        // We use an energy gate to mark silence vs "speech" in the synthetic fixture.
        // Use a stricter threshold so clearly different voices (or pitch-shifted voices) form distinct clusters.
        using OnlineSpeakerDiarizer diar = new(embeddingModelPath, sampleRateHz: 16000, windowMs: 1500, hopMs: 750, assignThreshold: 0.80f, hangoverMs: 250);
        using SpeakerEmbeddingExtractor embed = new(embeddingModelPath, sampleRateHz: 16000);

        HashSet<int> speakers = new();
        int segments = 0;
        diar.SegmentReady += (_, seg) =>
        {
            segments++;
            speakers.Add(seg.SpeakerId);
        };

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
        TimeSpan t = TimeSpan.Zero;

        int framesRead = 0;
        int speechFrames = 0;
        float maxRms = 0f;
        while (true)
        {
            int read = sample.Read(buf, 0, buf.Length);
            if (read < buf.Length)
            {
                break;
            }

            float[] chunk = new float[buf.Length];
            Array.Copy(buf, chunk, buf.Length);

            float rms = 0f;
            for (int i = 0; i < chunk.Length; i++)
            {
                rms += chunk[i] * chunk[i];
            }

            rms = MathF.Sqrt(rms / chunk.Length);
            maxRms = Math.Max(maxRms, rms);

            // Robust energy gate (fixture contains explicit silences).
            bool speech = rms >= 0.008f;
            if (speech)
            {
                speechFrames++;
            }

            diar.PushFrame(chunk, t, speech);
            t += TimeSpan.FromSeconds((double)frame / targetSr);
            framesRead++;

            // Keep this test bounded (~20 seconds max).
            if (framesRead > (int)(20 * targetSr / (double)frame))
            {
                break;
            }
        }

        diar.Flush(t);

        // Diagnostic: compare embeddings from early vs late windows to confirm the fixture is "separable".
        try
        {
            reader.Position = 0;
            ISampleProvider s2 = reader.ToSampleProvider();
            if (s2.WaveFormat.Channels == 2)
            {
                s2 = new StereoToMonoSampleProvider(s2) { LeftVolume = 0.5f, RightVolume = 0.5f };
            }
            if (s2.WaveFormat.SampleRate != targetSr)
            {
                s2 = new WdlResamplingSampleProvider(s2, targetSr);
            }

            int win = 16000 * 2; // 2s window
            float[] w1 = new float[win];
            float[] w2 = new float[win];
            _ = s2.Read(w1, 0, w1.Length);
            // skip ~2.5s (includes silence)
            float[] skip = new float[(int)(2.5 * targetSr)];
            _ = s2.Read(skip, 0, skip.Length);
            _ = s2.Read(w2, 0, w2.Length);

            float[] e1 = embed.GetEmbedding(w1);
            float[] e2 = embed.GetEmbedding(w2);
            float sim = Cosine(e1, e2);
            TestContext.Out.WriteLine($"Embedding cosine(sim early vs late)={sim:0.000} len={e1.Length}");
        }
        catch (Exception ex)
        {
            TestContext.Out.WriteLine($"Embedding diagnostic failed: {ex.GetType().Name}: {ex.Message}");
        }

        TestContext.Out.WriteLine(
            $"Diarization fixture frames={framesRead} speechFrames={speechFrames} maxRms={maxRms:0.000} segments={segments} speakers={string.Join(',', speakers.OrderBy(x => x))}");

        Assert.Multiple(() =>
        {
            Assert.That(framesRead, Is.GreaterThan(50), "Fixture too short or could not be read.");
            Assert.That(speechFrames, Is.GreaterThan(10), "Speech gate never activated; fixture generation or thresholds likely wrong.");
            Assert.That(segments, Is.GreaterThan(0), "Expected at least one diarization segment.");
            Assert.That(speakers, Has.Count.GreaterThanOrEqualTo(1), "Expected at least one diarization speaker ID.");
        });
    }

    private static float Cosine(float[] a, float[] b)
    {
        if (a.Length == 0 || b.Length == 0 || a.Length != b.Length)
        {
            return 1f;
        }

        float dot = 0f, magA = 0f, magB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }
        return magA == 0 || magB == 0 ? 1f : dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
    }
}
