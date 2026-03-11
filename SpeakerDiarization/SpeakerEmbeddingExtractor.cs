using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Dsp;

namespace SpeakerDiarization;

/// <summary>
/// ONNX Runtime wrapper for extracting distinct speaker embeddings (TitaNet/SpeakerNet) from audio chunks.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Translates raw audio blocks spanning multiple seconds into a high-dimensional dense vector 
/// (Speaker Embedding) that uniquely identifies the vocal tract characteristics of the speaker.
///
/// Responsibilities:
/// - Provide zero-copy loading of ONNX embedding models and manage hardware Session options.
/// - Implement Log-Mel filterbank extraction required by modern speech models.
/// - Convert raw PCM tensors or Mel feature tensors down into exactly one 1D Embedding float array.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime
/// - NAudio.Dsp (for Fast Fourier Transforms)
///
/// Architectural Role:
/// AI Service Component / Low-level wrapper.
/// </remarks>
public sealed class SpeakerEmbeddingExtractor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int[] _inputDims;
    private readonly string _outputName;
    private readonly int _sampleRateHz;

    /// <summary>
    /// Initializes the extractor and binds inputs/outputs by probing the loaded ONNX file.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX file.</param>
    /// <param name="sampleRateHz">The expected sample rate for mel-spectrogram generation.</param>
    public SpeakerEmbeddingExtractor(string modelPath, int sampleRateHz = 16000)
    {
        _sampleRateHz = sampleRateHz;
        _session = new InferenceSession(modelPath);

        // Assume single input.
        _inputName = _session.InputMetadata.Keys.First();
        _inputDims = _session.InputMetadata[_inputName].Dimensions;

        // Assume single output.
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Flushes native ONNX resources.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Feeds raw PCM into the ONNX session to retrieve the speaker vector. Automatically handles models requiring 1D vs 3D shapes.
    /// </summary>
    /// <param name="pcm16kMono">Raw 16kHz audio chunk.</param>
    /// <returns>A dense floating point array representing the voice characteristics.</returns>
    public float[] GetEmbedding(float[] pcm16kMono)
    {
        ArgumentNullException.ThrowIfNull(pcm16kMono);

        if (pcm16kMono.Length == 0)
        {
            return [];
        }

        // Decide based on input rank.
        if (_inputDims.Length == 2)
        {
            // Waveform: [1, N]
            DenseTensor<float> tensor = new([1, pcm16kMono.Length]);
            for (int i = 0; i < pcm16kMono.Length; i++)
            {
                tensor[0, i] = pcm16kMono[i];
            }

            List<NamedOnnxValue> inputs = new(1)
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            return To1DFloat(results.First(r => r.Name == _outputName).AsTensor<float>());
        }

        // Features: [1, F, T] where F is often 80.
        List<float[]> features = Fbank80(pcm16kMono, _sampleRateHz);
        if (features.Count == 0)
        {
            return [];
        }

        int F = features[0].Length;
        int T = features.Count;
        DenseTensor<float> featTensor = new([1, F, T]);
        for (int t = 0; t < T; t++)
        {
            float[] row = features[t];
            for (int f = 0; f < F; f++)
            {
                featTensor[0, f, t] = row[f];
            }
        }

        List<NamedOnnxValue> in2 = new(1)
        {
            NamedOnnxValue.CreateFromTensor(_inputName, featTensor)
        };
        using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(in2))
        {
            return To1DFloat(results.First(r => r.Name == _outputName).AsTensor<float>());
        }
    }

    private static float[] To1DFloat(Tensor<float> t)
    {
        float[] arr = new float[t.Length];
        int i = 0;
        foreach (float v in t)
        {
            arr[i++] = v;
        }

        return arr;
    }

    // Simple log-mel filterbank implementation.
    private static List<float[]> Fbank80(float[] pcm, int sr)
    {
        const int nFft = 512;
        const int win = 400; // 25ms @16k
        const int hop = 160; // 10ms @16k
        const int nMels = 80;
        const float fMin = 20f;
        float fMax = Math.Min(7600f, sr / 2f);

        if (pcm.Length < win)
        {
            return new List<float[]>();
        }

        float[][] melBank = BuildMelFilterBank(sr, nFft, nMels, fMin, fMax);
        float[] window = Hann(win);

        int frames = 1 + ((pcm.Length - win) / hop);
        List<float[]> feats = new(frames);

        Complex[] fftBuf = new Complex[nFft];
        float[] mag = new float[(nFft / 2) + 1];

        for (int i = 0; i < frames; i++)
        {
            int offset = i * hop;
            Array.Clear(fftBuf, 0, fftBuf.Length);

            for (int j = 0; j < win; j++)
            {
                float s = pcm[offset + j] * window[j];
                fftBuf[j].X = s;
                fftBuf[j].Y = 0;
            }

            FastFourierTransform.FFT(true, (int)Math.Log2(nFft), fftBuf);

            for (int k = 0; k < mag.Length; k++)
            {
                float re = fftBuf[k].X;
                float im = fftBuf[k].Y;
                mag[k] = (re * re) + (im * im);
            }

            float[] mel = new float[nMels];
            for (int m = 0; m < nMels; m++)
            {
                double sum = 0;
                float[] filt = melBank[m];
                for (int k = 0; k < mag.Length; k++)
                {
                    sum += filt[k] * mag[k];
                }

                mel[m] = (float)Math.Log10(Math.Max(1e-10, sum));
            }

            feats.Add(mel);
        }

        return feats;
    }

    private static float[] Hann(int n)
    {
        float[] w = new float[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = 0.5f - (0.5f * (float)Math.Cos(2 * Math.PI * i / (n - 1)));
        }
        return w;
    }

    private static float[][] BuildMelFilterBank(int sr, int nFft, int nMels, float fMin, float fMax)
    {
        int nBins = (nFft / 2) + 1;

        static double HzToMel(double hz)
        {
            return 2595.0 * Math.Log10(1.0 + (hz / 700.0));
        }

        static double MelToHz(double mel)
        {
            return 700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0);
        }

        double melMin = HzToMel(fMin);
        double melMax = HzToMel(fMax);

        double[] melPoints = new double[nMels + 2];
        for (int i = 0; i < melPoints.Length; i++)
        {
            melPoints[i] = melMin + ((melMax - melMin) * i / (nMels + 1));
        }

        double[] hzPoints = melPoints.Select(MelToHz).ToArray();
        int[] bin = hzPoints.Select(hz => (int)Math.Floor((nFft + 1) * hz / sr)).ToArray();

        float[][] bank = new float[nMels][];
        for (int m = 0; m < nMels; m++)
        {
            float[] f = new float[nBins];
            int left = Math.Clamp(bin[m], 0, nBins - 1);
            int center = Math.Clamp(bin[m + 1], 0, nBins - 1);
            int right = Math.Clamp(bin[m + 2], 0, nBins - 1);

            if (center == left)
            {
                center = Math.Min(left + 1, nBins - 1);
            }

            if (right == center)
            {
                right = Math.Min(center + 1, nBins - 1);
            }

            for (int k = left; k < center; k++)
            {
                f[k] = (float)(k - left) / Math.Max(1, center - left);
            }

            for (int k = center; k < right; k++)
            {
                f[k] = (float)(right - k) / Math.Max(1, right - center);
            }

            bank[m] = f;
        }

        return bank;
    }
}
