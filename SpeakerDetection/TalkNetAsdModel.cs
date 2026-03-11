using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Dsp;
using OpenCvSharp;

namespace SpeakerDetection;

/// <summary>
/// ONNX Runtime wrapper for TalkNet-style audio-visual active speaker detection.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Determines whether a person is actively speaking by fusing their cropped lip movements 
/// with the synchronous audio waveform using a convolutional architecture.
///
/// Responsibilities:
/// - Execute the ONNX mathematical operations for TalkNet ASD.
/// - Prepare visual tensors (112x112 grayscale face crops).
/// - Prepare audio tensors (compute MFCCs dynamically from raw PCM data).
/// - Return temporal speaking probability scores.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime
/// - OpenCvSharp (for matrix resizing)
///
/// Architectural Role:
/// AI Service Component / Multimodal Inference Model.
/// </remarks>
public sealed class TalkNetAsdModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _audioInput;
    private readonly string _visualInput;
    private readonly string _output;

    /// <summary>
    /// Initializes the ONNX inference session and binds input/output names.
    /// </summary>
    /// <param name="onnxPath">Absolute or relative file path to the TalkNet model.</param>
    public TalkNetAsdModel(string onnxPath)
    {
        _session = new InferenceSession(onnxPath);

        // Export script sets these names, but keep it robust.
        _audioInput = _session.InputMetadata.Keys.FirstOrDefault(k => k.Contains("audio", StringComparison.OrdinalIgnoreCase))
                      ?? _session.InputMetadata.Keys.First();
        _visualInput = _session.InputMetadata.Keys.FirstOrDefault(k => k.Contains("visual", StringComparison.OrdinalIgnoreCase))
                       ?? _session.InputMetadata.Keys.Skip(1).FirstOrDefault()
                       ?? _session.InputMetadata.Keys.First();
        _output = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Frees the unmanaged memory held by the ONNX InferenceSession.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Predicts per-frame speaking probability from a temporal window.
    /// </summary>
    /// <param name="audioPcm16kMono">Audio window samples (16kHz mono, [-1..1]).</param>
    /// <param name="visualGrayFrames">List of grayscale frames as Mats sized 112x112.</param>
    /// <param name="fps">Video fps used for MFCC alignment.</param>
    /// <returns>Speaking probabilities, one value per visual frame.</returns>
    public float[] Predict(float[] audioPcm16kMono, IReadOnlyList<Mat> visualGrayFrames, float fps = 25f)
    {
        ArgumentNullException.ThrowIfNull(visualGrayFrames);
        ArgumentNullException.ThrowIfNull(audioPcm16kMono);

        if (visualGrayFrames.Count == 0)
        {
            return [];
        }

        int tv = visualGrayFrames.Count;
        int ta = tv * 4;

        float[,] mfcc = Mfcc13(audioPcm16kMono, sampleRateHz: 16000, numFrames: ta, fps: fps);
        DenseTensor<float> audioTensor = new([1, ta, 13]);
        for (int t = 0; t < ta; t++)
        {
            for (int c = 0; c < 13; c++)
            {
                audioTensor[0, t, c] = mfcc[t, c];
            }
        }

        DenseTensor<float> visualTensor = new([1, tv, 112, 112]);
        for (int t = 0; t < tv; t++)
        {
            Mat src = visualGrayFrames[t];
            Mat? tmp = null;
            try
            {
                if (src.Width != 112 || src.Height != 112)
                {
                    tmp = new Mat();
                    Cv2.Resize(src, tmp, new Size(112, 112));
                    src = tmp;
                }

                for (int y = 0; y < 112; y++)
                {
                    for (int x = 0; x < 112; x++)
                    {
                        visualTensor[0, t, y, x] = src.At<byte>(y, x);
                    }
                }
            }
            finally
            {
                tmp?.Dispose();
            }
        }

        List<NamedOnnxValue> inputs = new()
        {
            NamedOnnxValue.CreateFromTensor(_audioInput, audioTensor),
            NamedOnnxValue.CreateFromTensor(_visualInput, visualTensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        Tensor<float> outTensor = results.First(r => r.Name == _output).AsTensor<float>();
        float[] outArr = new float[outTensor.Length];
        int i = 0;
        foreach (float v in outTensor)
        {
            outArr[i++] = v;
        }

        return outArr;
    }

    // Basic MFCC (python_speech_features-like defaults; enough to drive the model).
    private static float[,] Mfcc13(float[] pcm, int sampleRateHz, int numFrames, float fps)
    {
        // Win/step scaled by fps to align audio/visual.
        double winLen = 0.025 * 25.0 / Math.Max(1.0, fps);
        double winStep = 0.010 * 25.0 / Math.Max(1.0, fps);
        int win = Math.Max(200, (int)Math.Round(winLen * sampleRateHz));
        int hop = Math.Max(80, (int)Math.Round(winStep * sampleRateHz));
        int nFft = 512;
        int nMels = 26;
        int nCep = 13;

        if (pcm.Length < win)
        {
            pcm = PadWrap(pcm, win);
        }

        int frames = 1 + ((pcm.Length - win) / hop);
        float[][] filterbank = BuildMelFilterBank(sampleRateHz, nFft, nMels, 0, sampleRateHz / 2);
        float[] window = Hann(win);

        float[,] mfcc = new float[numFrames, nCep];

        Complex[] fftBuf = new Complex[nFft];
        float[] mag = new float[(nFft / 2) + 1];
        double[] melE = new double[nMels];

        int outT = 0;
        for (int i = 0; i < frames && outT < numFrames; i++)
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

            for (int m = 0; m < nMels; m++)
            {
                double sum = 0;
                float[] filt = filterbank[m];
                for (int k = 0; k < mag.Length; k++)
                {
                    sum += filt[k] * mag[k];
                }

                melE[m] = Math.Log(Math.Max(1e-10, sum));
            }

            double[] cep = Dct(melE, nCep);
            for (int c = 0; c < nCep; c++)
            {
                mfcc[outT, c] = (float)cep[c];
            }

            outT++;
        }

        // Pad/wrap to numFrames
        if (outT < numFrames && outT > 0)
        {
            int t = outT;
            while (t < numFrames)
            {
                for (int c = 0; c < nCep; c++)
                {
                    mfcc[t, c] = mfcc[t % outT, c];
                }

                t++;
            }
        }

        return mfcc;
    }

    private static float[] PadWrap(float[] x, int n)
    {
        float[] y = new float[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = x[i % x.Length];
        }

        return y;
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

    private static double[] Dct(double[] x, int nCep)
    {
        int N = x.Length;
        double[] y = new double[nCep];
        for (int k = 0; k < nCep; k++)
        {
            double sum = 0;
            for (int n = 0; n < N; n++)
            {
                sum += x[n] * Math.Cos(Math.PI * k * ((2 * n) + 1) / (2 * N));
            }

            y[k] = sum;
        }
        return y;
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
