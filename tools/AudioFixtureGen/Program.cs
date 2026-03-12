using System.Speech.Synthesis;
using System.Globalization;

namespace AudioFixtureGen;

internal static class Program
{
    public static int Main(string[] args)
    {
        bool twoSpeakers = args.Any(a => string.Equals(a, "--two-speakers", StringComparison.OrdinalIgnoreCase));
        bool twoSpeakersSynth = args.Any(a => string.Equals(a, "--two-speakers-synth", StringComparison.OrdinalIgnoreCase));

        string outPath = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal)
            ? args[0]
            : Path.Combine("E2ETests", "audio", (twoSpeakers || twoSpeakersSynth) ? "e2e_fixture_two_speakers.wav" : "e2e_fixture_10_words.wav");

        string text = args.Length > 1
            ? string.Join(' ', args.Skip(1).Where(a => !string.Equals(a, "--two-speakers", StringComparison.OrdinalIgnoreCase)))
            : "This offline test audio contains more than ten words for transcription.";

        string fullOut = Path.GetFullPath(outPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullOut)!);

        if (File.Exists(fullOut))
        {
            File.Delete(fullOut);
        }

        if (twoSpeakersSynth)
        {
            WriteSyntheticTwoSpeakersWav(fullOut);
        }
        else
        {
            using SpeechSynthesizer synth = new();
            synth.Rate = 0;
            synth.Volume = 100;
            synth.SetOutputToWaveFile(fullOut);

            if (!twoSpeakers)
            {
                synth.Speak(text);
            }
            else
            {
                // Build a deterministic "two speakers" fixture by switching installed SAPI voices.
                // If only one voice exists, fall back to a synthetic two-speaker signal.
                var voices = synth.GetInstalledVoices(CultureInfo.CurrentUICulture)
                    .Where(v => v.Enabled)
                    .Select(v => v.VoiceInfo.Name)
                    .OrderBy(v => v, StringComparer.OrdinalIgnoreCase)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();

                if (voices.Count < 2)
                {
                    synth.SetOutputToNull();
                    WriteSyntheticTwoSpeakersWav(fullOut);
                }
                else
                {
                    string v1 = voices[0];
                    string v2 = voices[1];

                    PromptBuilder pb = new();
                    pb.StartVoice(v1);
                    pb.AppendText("Speaker one says: we validate diarization with two distinct offline speakers for testing.");
                    pb.EndVoice();
                    pb.AppendBreak(TimeSpan.FromMilliseconds(350));

                    pb.StartVoice(v2);
                    pb.AppendText("Speaker two says: this second voice should sound different and separate clusters reliably.");
                    pb.EndVoice();
                    pb.AppendBreak(TimeSpan.FromMilliseconds(350));

                    pb.StartVoice(v1);
                    pb.AppendText("Speaker one continues: this is a repeat segment to strengthen the clustering evidence.");
                    pb.EndVoice();
                    pb.AppendBreak(TimeSpan.FromMilliseconds(350));

                    pb.StartVoice(v2);
                    pb.AppendText("Speaker two continues: we alternate speakers with short pauses to mimic turn taking.");
                    pb.EndVoice();

                    synth.Speak(pb);
                }
            }
        }

        Console.WriteLine($"Wrote: {fullOut}");
        Console.WriteLine($"Text: {text}");
        return 0;
    }

    private static void WriteSyntheticTwoSpeakersWav(string outPath)
    {
        const int sr = 16000;

        // Preferred: derive two "different speakers" from the base speech fixture by strong time/pitch warp,
        // then alternate segments with silence. This is deterministic and speech-like enough for speaker-embedding models.
        // Fallback: synthetic tone/noise if the base fixture is not available.
        string baseFixture = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(outPath) ?? ".", "e2e_fixture_10_words.wav"));

        float[] a;
        float[] b;
        if (File.Exists(baseFixture))
        {
            float[] basePcm = ReadWavToMonoFloat(baseFixture);
            float[] base16 = ResampleLinear(basePcm, DetectSampleRateHz(baseFixture), sr);

            // Take two different 2.0s slices (different content) to help separate embeddings.
            int startA = Math.Min(base16.Length - 1, (int)(0.6 * sr));
            int startB = Math.Min(base16.Length - 1, (int)(2.8 * sr));
            int len = (int)(2.0 * sr);

            a = base16.Skip(startA).Take(Math.Min(len, base16.Length - startA)).ToArray();
            if (a.Length < (int)(2.0 * sr))
            {
                a = a.Concat(new float[(int)(2.0 * sr) - a.Length]).ToArray();
            }

            float[] b0 = base16.Skip(startB).Take(Math.Min(len, base16.Length - startB)).ToArray();
            if (b0.Length < len)
            {
                b0 = b0.Concat(new float[len - b0.Length]).ToArray();
            }

            // Create b by pitch/speed warp + strong spectral/temporal effects, then tile to full length.
            float[] warped = ResampleByFactor(b0, factor: 2.2);
            b = TileToLength(warped, len);

            // Low-pass a slightly, pre-emphasize b (different spectral envelope).
            a = MovingAverageLowPass(a, taps: 5);
            b = PreEmphasis(b, alpha: 0.97f);
            b = ApplyAmplitudeModulation(b, sr, hz: 5.5f, depth: 0.35f);
            b = AddDeterministicNoise(b, seed: 777, scale: 0.015f);
            b = SoftClip(b, drive: 2.2f);
        }
        else
        {
            a = SynthSpeaker(sr, seconds: 2.0, toneHz: 220f, noiseScale: 0.08f, seed: 123);
            b = SynthSpeaker(sr, seconds: 2.0, toneHz: 880f, noiseScale: 0.08f, seed: 456);
        }

        float[] silence = new float[(int)(0.35 * sr)];

        List<float> pcm = new();
        for (int i = 0; i < 4; i++)
        {
            pcm.AddRange(a);
            pcm.AddRange(silence);
            pcm.AddRange(b);
            pcm.AddRange(silence);
        }

        WriteWav16Mono(outPath, sr, pcm);
    }

    private static float[] TileToLength(float[] x, int targetLen)
    {
        if (targetLen <= 0)
        {
            return [];
        }

        if (x.Length == 0)
        {
            return new float[targetLen];
        }

        float[] y = new float[targetLen];
        for (int i = 0; i < y.Length; i++)
        {
            y[i] = x[i % x.Length];
        }
        return y;
    }

    private static float[] ApplyAmplitudeModulation(float[] x, int sr, float hz, float depth)
    {
        if (x.Length == 0 || sr <= 0)
        {
            return x;
        }

        depth = Math.Clamp(depth, 0f, 0.95f);
        double step = 2.0 * Math.PI * Math.Clamp(hz, 0.2f, 12f) / sr;
        double phase = 0.0;

        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float m = 1f + (depth * (float)Math.Sin(phase));
            phase += step;
            y[i] = x[i] * m;
        }
        return y;
    }

    private static float[] AddDeterministicNoise(float[] x, int seed, float scale)
    {
        if (x.Length == 0 || scale <= 0f)
        {
            return x;
        }

        scale = Math.Clamp(scale, 0f, 0.2f);
        Random rng = new(seed);
        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float n = (float)((rng.NextDouble() * 2.0) - 1.0);
            y[i] = x[i] + (scale * n);
        }
        return y;
    }

    private static float[] SoftClip(float[] x, float drive)
    {
        if (x.Length == 0)
        {
            return x;
        }

        drive = Math.Clamp(drive, 0.5f, 6.0f);
        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i] * drive;
            // tanh soft clip
            float c = (float)Math.Tanh(v);
            y[i] = Math.Clamp(c, -0.98f, 0.98f);
        }
        return y;
    }

    private static float[] MovingAverageLowPass(float[] x, int taps)
    {
        taps = Math.Clamp(taps, 1, 31);
        if (taps == 1 || x.Length == 0)
        {
            return x;
        }

        float[] y = new float[x.Length];
        int r = taps / 2;
        for (int i = 0; i < x.Length; i++)
        {
            double sum = 0;
            int count = 0;
            for (int k = -r; k <= r; k++)
            {
                int j = i + k;
                if ((uint)j < (uint)x.Length)
                {
                    sum += x[j];
                    count++;
                }
            }
            y[i] = count > 0 ? (float)(sum / count) : x[i];
        }
        return y;
    }

    private static float[] PreEmphasis(float[] x, float alpha)
    {
        if (x.Length == 0)
        {
            return x;
        }

        alpha = Math.Clamp(alpha, 0.0f, 0.99f);
        float[] y = new float[x.Length];
        y[0] = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            y[i] = x[i] - (alpha * x[i - 1]);
        }
        return y;
    }

    private static int DetectSampleRateHz(string wavPath)
    {
        using FileStream fs = File.OpenRead(wavPath);
        using BinaryReader br = new(fs);
        string riff = new(br.ReadChars(4));
        if (!string.Equals(riff, "RIFF", StringComparison.Ordinal))
        {
            return 16000;
        }
        _ = br.ReadInt32();
        _ = new string(br.ReadChars(4)); // WAVE

        // Find "fmt " chunk
        while (fs.Position + 8 <= fs.Length)
        {
            string chunkId = new(br.ReadChars(4));
            int chunkSize = br.ReadInt32();
            if (string.Equals(chunkId, "fmt ", StringComparison.Ordinal))
            {
                _ = br.ReadInt16(); // format
                _ = br.ReadInt16(); // channels
                int sr = br.ReadInt32();
                return sr > 0 ? sr : 16000;
            }

            fs.Position += chunkSize;
        }

        return 16000;
    }

    private static float[] ReadWavToMonoFloat(string wavPath)
    {
        using FileStream fs = File.OpenRead(wavPath);
        using BinaryReader br = new(fs);
        string riff = new(br.ReadChars(4));
        if (!string.Equals(riff, "RIFF", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Not a RIFF file.");
        }
        _ = br.ReadInt32();
        string wave = new(br.ReadChars(4));
        if (!string.Equals(wave, "WAVE", StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Not a WAVE file.");
        }

        short channels = 1;
        short bits = 16;

        // Parse chunks
        byte[]? data = null;
        while (fs.Position + 8 <= fs.Length)
        {
            string chunkId = new(br.ReadChars(4));
            int chunkSize = br.ReadInt32();
            if (string.Equals(chunkId, "fmt ", StringComparison.Ordinal))
            {
                short fmt = br.ReadInt16();
                channels = br.ReadInt16();
                _ = br.ReadInt32(); // sr
                _ = br.ReadInt32(); // byte rate
                _ = br.ReadInt16(); // align
                bits = br.ReadInt16();

                // Skip any remaining fmt bytes
                int remain = chunkSize - 16;
                if (remain > 0)
                {
                    fs.Position += remain;
                }

                if (fmt != 1 || bits != 16)
                {
                    throw new InvalidOperationException("Only PCM 16-bit WAV is supported for fixture derivation.");
                }
            }
            else if (string.Equals(chunkId, "data", StringComparison.Ordinal))
            {
                data = br.ReadBytes(chunkSize);
                break;
            }
            else
            {
                fs.Position += chunkSize;
            }
        }

        if (data == null)
        {
            throw new InvalidOperationException("WAV data chunk not found.");
        }

        int samples = data.Length / 2 / Math.Max(1, (int)channels);
        float[] pcm = new float[samples];
        int idx = 0;
        for (int i = 0; i < samples; i++)
        {
            int off = i * channels * 2;
            short s = BitConverter.ToInt16(data, off);
            pcm[idx++] = s / (float)short.MaxValue;
        }

        return pcm;
    }

    private static float[] ResampleLinear(float[] input, int inSr, int outSr)
    {
        if (input.Length == 0 || inSr <= 0 || outSr <= 0 || inSr == outSr)
        {
            return input;
        }

        double factor = outSr / (double)inSr;
        int outLen = Math.Max(1, (int)Math.Round(input.Length * factor));
        float[] output = new float[outLen];

        for (int i = 0; i < outLen; i++)
        {
            double src = i / factor;
            int i0 = (int)Math.Floor(src);
            int i1 = Math.Min(input.Length - 1, i0 + 1);
            double t = src - i0;
            float a = input[Math.Clamp(i0, 0, input.Length - 1)];
            float b = input[i1];
            output[i] = (float)((a * (1.0 - t)) + (b * t));
        }

        return output;
    }

    private static float[] ResampleByFactor(float[] input, double factor)
    {
        if (input.Length == 0)
        {
            return [];
        }

        factor = Math.Clamp(factor, 0.5, 3.0);
        int outLen = Math.Max(1, (int)Math.Round(input.Length / factor));
        float[] output = new float[outLen];

        for (int i = 0; i < outLen; i++)
        {
            double src = i * factor;
            int i0 = (int)Math.Floor(src);
            int i1 = Math.Min(input.Length - 1, i0 + 1);
            double t = src - i0;

            float a = input[Math.Clamp(i0, 0, input.Length - 1)];
            float b = input[i1];
            output[i] = (float)((a * (1.0 - t)) + (b * t));
        }

        return output;
    }

    private static float[] SynthSpeaker(int sr, double seconds, float toneHz, float noiseScale, int seed)
    {
        int n = Math.Max(1, (int)Math.Round(sr * Math.Clamp(seconds, 0.2, 30.0)));
        float[] x = new float[n];
        Random rng = new(seed);

        double phase = 0;
        double step = 2.0 * Math.PI * toneHz / sr;
        for (int i = 0; i < n; i++)
        {
            float tone = (float)Math.Sin(phase);
            phase += step;

            // deterministic noise in [-1,1]
            float noise = (float)((rng.NextDouble() * 2.0) - 1.0);

            // Simple amplitude envelope to look more like speech bursts.
            double t = i / (double)n;
            float env = (float)Math.Clamp(Math.Sin(Math.PI * t), 0.0, 1.0);
            float s = (0.12f * tone) + (noiseScale * noise);
            x[i] = env * s;
        }

        // Normalize
        float max = 1e-6f;
        foreach (float v in x)
        {
            max = Math.Max(max, Math.Abs(v));
        }
        float scale = max > 0.95f ? 0.95f / max : 1f;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= scale;
        }

        return x;
    }

    private static void WriteWav16Mono(string outPath, int sr, List<float> pcm)
    {
        int dataBytes = pcm.Count * 2;

        using FileStream fs = File.Create(outPath);
        using BinaryWriter bw = new(fs);

        // RIFF header
        bw.Write("RIFF"u8.ToArray());
        bw.Write(36 + dataBytes);
        bw.Write("WAVE"u8.ToArray());

        // fmt chunk
        bw.Write("fmt "u8.ToArray());
        bw.Write(16);
        bw.Write((short)1); // PCM
        bw.Write((short)1); // mono
        bw.Write(sr);
        bw.Write(sr * 2); // byte rate
        bw.Write((short)2); // block align
        bw.Write((short)16); // bits

        // data chunk
        bw.Write("data"u8.ToArray());
        bw.Write(dataBytes);

        foreach (float f in pcm)
        {
            short s = (short)Math.Clamp((int)MathF.Round(f * short.MaxValue), short.MinValue, short.MaxValue);
            bw.Write(s);
        }
    }
}
