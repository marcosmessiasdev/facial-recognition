using System;
using System.Collections.Generic;
using System.Linq;
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

    /// <summary>
    /// Initializes a new instance of the SileroVad class with the specified model.
    /// </summary>
    /// <param name="modelPath">The absolute path to the silero_vad.onnx model.</param>
    public SileroVad(string modelPath)
    {
        _session = new InferenceSession(modelPath, new SessionOptions());

        (_audioInputName, _srInputName, _hInputName, _cInputName) = SelectInputs(_session.InputMetadata);
        (_probOutputName, _hOutputName, _cOutputName) = SelectOutputs(_session.OutputMetadata);

        if (_hInputName != null)
            _h = CreateZeroState(_session.InputMetadata[_hInputName].Dimensions);
        if (_cInputName != null)
            _c = CreateZeroState(_session.InputMetadata[_cInputName].Dimensions);
    }

    /// <summary>
    /// Evaluates the probability of human speech in the provided mono samples.
    /// </summary>
    /// <param name="monoSamples">Input audio samples normalized to [-1..1].</param>
    /// <param name="sampleRateHz">The sample rate of the input audio (usually 16000).</param>
    /// <returns>A probability score between 0.0 and 1.0.</returns>
    public float GetSpeechProbability(float[] monoSamples, int sampleRateHz)
    {
        if (monoSamples.Length == 0) return 0f;

        var inputs = new List<NamedOnnxValue>();

        var audio = new DenseTensor<float>(new[] { 1, monoSamples.Length });
        for (int i = 0; i < monoSamples.Length; i++)
            audio[0, i] = monoSamples[i];
        inputs.Add(NamedOnnxValue.CreateFromTensor(_audioInputName, audio));

        if (_srInputName != null)
        {
            var sr = new DenseTensor<long>(new[] { 1 });
            sr[0] = sampleRateHz;
            inputs.Add(NamedOnnxValue.CreateFromTensor(_srInputName, sr));
        }

        if (_hInputName != null && _h != null)
            inputs.Add(NamedOnnxValue.CreateFromTensor(_hInputName, _h));
        if (_cInputName != null && _c != null)
            inputs.Add(NamedOnnxValue.CreateFromTensor(_cInputName, _c));

        using var results = _session.Run(inputs);

        float prob = results.First(r => r.Name == _probOutputName).AsEnumerable<float>().FirstOrDefault();

        if (_hOutputName != null && _h != null)
        {
            var nextH = results.FirstOrDefault(r => r.Name == _hOutputName)?.AsTensor<float>();
            if (nextH != null) _h = Clone(nextH);
        }

        if (_cOutputName != null && _c != null)
        {
            var nextC = results.FirstOrDefault(r => r.Name == _cOutputName)?.AsTensor<float>();
            if (nextC != null) _c = Clone(nextC);
        }

        return prob;
    }

    /// <summary>
    /// Resets the internal LSTM state tensors to zero.
    /// </summary>
    public void ResetState()
    {
        if (_hInputName != null)
            _h = CreateZeroState(_session.InputMetadata[_hInputName].Dimensions);
        if (_cInputName != null)
            _c = CreateZeroState(_session.InputMetadata[_cInputName].Dimensions);
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

        foreach (var kvp in inputs)
        {
            var name = kvp.Key;
            var meta = kvp.Value;
            var dims = meta.Dimensions;

            if (meta.ElementType == typeof(float) && dims.Length == 2 && (dims[0] == 1 || dims[0] <= 0))
                audio ??= name;
            else if (meta.ElementType == typeof(long))
                sr ??= name;
            else if (meta.ElementType == typeof(float) && dims.Length == 3)
            {
                // LSTM state tensors usually look like [2, 1, 64] or similar.
                if (h == null) h = name;
                else if (c == null) c = name;
            }
        }

        if (audio == null)
            throw new InvalidOperationException("Silero VAD input not recognized: expected a float audio tensor input [1, T].");

        return (audio, sr, h, c);
    }

    private static (string prob, string? hOut, string? cOut) SelectOutputs(IReadOnlyDictionary<string, NodeMetadata> outputs)
    {
        string? prob = null;
        string? h = null;
        string? c = null;

        foreach (var kvp in outputs)
        {
            var name = kvp.Key;
            var meta = kvp.Value;
            var dims = meta.Dimensions;

            if (meta.ElementType == typeof(float) && dims.Length == 2 && (dims[0] == 1 || dims[0] <= 0))
                prob ??= name;
            else if (meta.ElementType == typeof(float) && dims.Length == 3)
            {
                if (h == null) h = name;
                else if (c == null) c = name;
            }
        }

        prob ??= outputs.Keys.First();
        return (prob, h, c);
    }

    private static DenseTensor<float> CreateZeroState(int[] dims)
    {
        var normalized = dims.Select(d => d <= 0 ? 1 : d).ToArray();
        return new DenseTensor<float>(normalized);
    }

    private static DenseTensor<float> Clone(Tensor<float> t)
    {
        var dims = t.Dimensions.ToArray();
        var clone = new DenseTensor<float>(dims);
        int i = 0;
        foreach (var v in t)
            clone.Buffer.Span[i++] = v;
        return clone;
    }
}


