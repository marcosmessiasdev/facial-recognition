using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AudioProcessing;

/// <summary>
/// Implements Voice Activity Detection (VAD) using the Silero ONNX model.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Determines if human speech is present in an audio stream to enable context-aware analysis.
///
/// Responsibilities:
/// - Load and manage the Silero VAD model via ONNX Runtime.
/// - Maintain the recurrent state (LSTM H and C tensors) across sequential audio chunks.
/// - Perform inference on raw audio samples and return speech probability.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime (Inference Engine)
///
/// Architectural Role:
/// Infrastructure Component / Audio Processing Service.
///
/// Constraints:
/// - Expects mono audio at specific sample rates (8k or 16k usually).
/// - stateful: requires sequential processing of audio frames for accurate results.
/// </remarks>
public sealed class SileroVad : IDisposable
{
    private readonly InferenceSession _session;

    private readonly string _audioInputName;
    private readonly string? _srInputName;
    private readonly string? _hInputName;
    private readonly string? _cInputName;

    private readonly string _probOutputName;
    private readonly string? _hOutputName;
    private readonly string? _cOutputName;

    private DenseTensor<float>? _h;
    private DenseTensor<float>? _c;

    public string DebugInfo { get; }

    /// <summary>
    /// Initializes a new instance of the SileroVad class with the specified model.
    /// </summary>
    /// <param name="modelPath">The absolute path to the silero_vad.onnx model.</param>
    public SileroVad(string modelPath)
    {
        using SessionOptions options = new();
        _session = new InferenceSession(modelPath, options);

        (_audioInputName, _srInputName, _hInputName, _cInputName) = SelectInputs(_session.InputMetadata);
        (_probOutputName, _hOutputName, _cOutputName) = SelectOutputs(_session.OutputMetadata);

        DebugInfo = BuildDebugInfo(
            _session,
            _audioInputName,
            _srInputName,
            _hInputName,
            _cInputName,
            _probOutputName,
            _hOutputName,
            _cOutputName);

        if (_hInputName != null)
        {
            _h = CreateZeroState(_session.InputMetadata[_hInputName].Dimensions);
        }

        if (_cInputName != null)
        {
            _c = CreateZeroState(_session.InputMetadata[_cInputName].Dimensions);
        }
    }

    /// <summary>
    /// Evaluates the probability of human speech in the provided mono samples.
    /// </summary>
    /// <param name="monoSamples">Input audio samples normalized to [-1..1].</param>
    /// <param name="sampleRateHz">The sample rate of the input audio (usually 16000).</param>
    /// <returns>A probability score between 0.0 and 1.0.</returns>
    public float GetSpeechProbability(float[] monoSamples, int sampleRateHz)
    {
        ArgumentNullException.ThrowIfNull(monoSamples);

        if (monoSamples.Length == 0)
        {
            return 0f;
        }

        List<NamedOnnxValue> inputs = new();

        DenseTensor<float> audio = new([1, monoSamples.Length]);
        for (int i = 0; i < monoSamples.Length; i++)
        {
            audio[0, i] = monoSamples[i];
        }

        inputs.Add(NamedOnnxValue.CreateFromTensor(_audioInputName, audio));

        if (_srInputName != null)
        {
            // Some models define sample-rate as a scalar (rank-0) tensor.
            DenseTensor<long> sr = new([]);
            sr.Buffer.Span[0] = sampleRateHz;
            inputs.Add(NamedOnnxValue.CreateFromTensor(_srInputName, sr));
        }

        if (_hInputName != null && _h != null)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(_hInputName, _h));
        }

        if (_cInputName != null && _c != null)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(_cInputName, _c));
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

        // Some exported variants return a sequence-like probability output (e.g. [N,1]).
        // For streaming VAD we want the most recent probability in the chunk.
        float prob = 0f;
        Tensor<float> probTensor = results.First(r => r.Name == _probOutputName).AsTensor<float>();
        foreach (float v in probTensor)
        {
            prob = v;
        }

        if (_hOutputName != null && _h != null)
        {
            Tensor<float>? nextH = results.FirstOrDefault(r => r.Name == _hOutputName)?.AsTensor<float>();
            if (nextH != null)
            {
                _h = Clone(nextH);
            }
        }

        if (_cOutputName != null && _c != null)
        {
            Tensor<float>? nextC = results.FirstOrDefault(r => r.Name == _cOutputName)?.AsTensor<float>();
            if (nextC != null)
            {
                _c = Clone(nextC);
            }
        }

        return prob;
    }

    /// <summary>
    /// Resets the internal LSTM state tensors to zero.
    /// </summary>
    public void ResetState()
    {
        if (_hInputName != null)
        {
            _h = CreateZeroState(_session.InputMetadata[_hInputName].Dimensions);
        }

        if (_cInputName != null)
        {
            _c = CreateZeroState(_session.InputMetadata[_cInputName].Dimensions);
        }
    }

    /// <summary>
    /// Releases the ONNX runtime session resources.
    /// </summary>
    public void Dispose() => _session.Dispose();

    private static (string audio, string? sr, string? h, string? c) SelectInputs(IReadOnlyDictionary<string, NodeMetadata> inputs)
    {
        string? audio = null;
        string? sr = null;
        string? h = null;
        string? c = null;

        foreach (KeyValuePair<string, NodeMetadata> kvp in inputs)
        {
            string name = kvp.Key;
            NodeMetadata meta = kvp.Value;
            int[] dims = meta.Dimensions;

            if (meta.ElementType == typeof(float) && dims.Length == 2 && (dims[0] == 1 || dims[0] <= 0))
            {
                audio ??= name;
            }
            else if (meta.ElementType == typeof(long))
            {
                sr ??= name;
            }
            else if (meta.ElementType == typeof(float) && dims.Length == 3)
            {
                // LSTM state tensors usually look like [2, 1, 64] or similar.
                if (h == null)
                {
                    h = name;
                }
                else
                {
                    c ??= name;
                }
            }
        }

        return audio == null
            ? throw new InvalidOperationException("Silero VAD input not recognized: expected a float audio tensor input [1, T].")
            : (audio, sr, h, c);
    }

    private static (string prob, string? hOut, string? cOut) SelectOutputs(IReadOnlyDictionary<string, NodeMetadata> outputs)
    {
        string? prob = null;
        string? h = null;
        string? c = null;

        foreach (KeyValuePair<string, NodeMetadata> kvp in outputs)
        {
            string name = kvp.Key;
            NodeMetadata meta = kvp.Value;
            int[] dims = meta.Dimensions;

            // Silero VAD probability output is typically a scalar-ish float tensor (e.g. [1] or [1,1]).
            if (meta.ElementType == typeof(float) && dims.Length <= 2)
            {
                prob ??= name;
            }
            else if (meta.ElementType == typeof(float) && dims.Length == 3)
            {
                if (h == null)
                {
                    h = name;
                }
                else
                {
                    c ??= name;
                }
            }
        }

        prob ??= outputs.Keys.First();
        return (prob, h, c);
    }

    private static DenseTensor<float> CreateZeroState(int[] dims)
    {
        int[] normalized = [.. dims.Select(d => d <= 0 ? 1 : d)];
        return new DenseTensor<float>(normalized);
    }

    private static DenseTensor<float> Clone(Tensor<float> t)
    {
        int[] dims = t.Dimensions.ToArray();
        DenseTensor<float> clone = new(dims);
        int i = 0;
        foreach (float v in t)
        {
            clone.Buffer.Span[i++] = v;
        }

        return clone;
    }

    private static string BuildDebugInfo(
        InferenceSession session,
        string audioIn,
        string? srIn,
        string? hIn,
        string? cIn,
        string probOut,
        string? hOut,
        string? cOut)
    {
        static string Dims(int[] d)
        {
            return d.Length == 0 ? "[]" : $"[{string.Join(",", d)}]";
        }

        string audioDims = Dims(session.InputMetadata[audioIn].Dimensions);
        string srDims = srIn != null ? Dims(session.InputMetadata[srIn].Dimensions) : "n/a";
        string hDims = hIn != null ? Dims(session.InputMetadata[hIn].Dimensions) : "n/a";
        string cDims = cIn != null ? Dims(session.InputMetadata[cIn].Dimensions) : "n/a";

        string pDims = Dims(session.OutputMetadata[probOut].Dimensions);
        string hODims = hOut != null ? Dims(session.OutputMetadata[hOut].Dimensions) : "n/a";
        string cODims = cOut != null ? Dims(session.OutputMetadata[cOut].Dimensions) : "n/a";

        return $"inputs: audio='{audioIn}'{audioDims} sr='{srIn ?? "n/a"}'{srDims} h='{hIn ?? "n/a"}'{hDims} c='{cIn ?? "n/a"}'{cDims} | outputs: prob='{probOut}'{pDims} h='{hOut ?? "n/a"}'{hODims} c='{cOut ?? "n/a"}'{cODims}";
    }
}
