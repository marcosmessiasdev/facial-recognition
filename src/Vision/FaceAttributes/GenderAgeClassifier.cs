using System.Globalization;
using FacialRecognition.Domain;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceAttributes;

/// <summary>
/// Predicts gender and age from facial features using a multi-output ONNX model.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Provides efficient attribute extraction by running a single inference session to predict 
/// both gender and age categories, reducing overall computational overhead.
///
/// Responsibilities:
/// - Load and manage the lifecycle of a multi-output ONNX model.
/// - Crop and align faces to match the model's required input resolution.
/// - Execute inference and post-process logits into human-readable labels and confidence scores.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime (Inference Engine)
/// - OpenCvSharp (Image preprocessing and alignment)
///
/// Architectural Role:
/// Infrastructure Component / Analysis Service.
///
/// Constraints:
/// - Requires faces to be reasonably centered and aligned for optimal accuracy.
/// </remarks>
public sealed class GenderAgeClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly int _inputSize;
    private readonly bool _nhwc;

    /// <summary>
    /// Initializes a new instance of the GenderAgeClassifier with the specified ONNX model.
    /// </summary>
    /// <param name="modelPath">The absolute path to the .onnx model file.</param>
    public GenderAgeClassifier(string modelPath)
    {
        using SessionOptions options = new();
        _session = new InferenceSession(modelPath, options);
        (_inputName, _inputSize, _nhwc) = SelectImageInput(_session.InputMetadata);
        _outputName = SelectOutput(_session.OutputMetadata);
    }

    /// <summary>
    /// Predicts the gender and age of a face within an image.
    /// </summary>
    /// <param name="frameBgr">The original BGR image frame.</param>
    /// <param name="faceBox">The bounding box of the face to analyze.</param>
    /// <returns>A tuple containing the gender label, confidence, and age label, or null if analysis fails.</returns>
    public (string genderLabel, float genderConfidence, string ageLabel)? Predict(Mat frameBgr, BoundingBox faceBox)
    {
        ArgumentNullException.ThrowIfNull(frameBgr);
        ArgumentNullException.ThrowIfNull(faceBox);

        if (frameBgr.Empty())
        {
            return null;
        }

        using Mat aligned = AlignFace(frameBgr, faceBox, _inputSize);

        using Mat rgb = new();
        if (aligned.Channels() == 4)
        {
            Cv2.CvtColor(aligned, rgb, ColorConversionCodes.BGRA2RGB);
        }
        else
        {
            Cv2.CvtColor(aligned, rgb, ColorConversionCodes.BGR2RGB);
        }

        DenseTensor<float> tensor = _nhwc
            ? new DenseTensor<float>([1, _inputSize, _inputSize, 3])
            : new DenseTensor<float>([1, 3, _inputSize, _inputSize]);

        unsafe
        {
            byte* ptr = rgb.DataPointer;
            int stride = (int)rgb.Step();

            for (int y = 0; y < _inputSize; y++)
            {
                for (int x = 0; x < _inputSize; x++)
                {
                    int idx = (y * stride) + (x * 3);
                    float r = ptr[idx + 0];
                    float g = ptr[idx + 1];
                    float b = ptr[idx + 2];

                    if (_nhwc)
                    {
                        tensor[0, y, x, 0] = r;
                        tensor[0, y, x, 1] = g;
                        tensor[0, y, x, 2] = b;
                    }
                    else
                    {
                        tensor[0, 0, y, x] = r;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = b;
                    }
                }
            }
        }

        List<NamedOnnxValue> inputs = new()
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        float[] output = [.. results.First(r => r.Name == _outputName).AsEnumerable<float>()];

        if (output.Length < 3)
        {
            return null;
        }

        // Based on upstream yakhyo/facial-analysis behavior:
        //   output[0..1] => gender logits
        //   output[2]    => age normalized [0..1], age = output[2] * 100
        (int genderIndex, float genderConfidence) = SoftmaxArgMax(output[0], output[1]);
        string genderLabel = genderIndex == 1 ? "Male" : "Female";

        int ageYears = (int)Math.Round(output[2] * 100.0);
        ageYears = Math.Clamp(ageYears, 0, 100);
        string ageLabel = ageYears.ToString(CultureInfo.InvariantCulture);

        return (genderLabel, genderConfidence, ageLabel);
    }

    /// <summary>
    /// Releases the ONNX runtime session resources.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Computes softmax for two logits and returns the winning index and probability.
    /// </summary>
    private static (int index, float confidence) SoftmaxArgMax(float a, float b)
    {
        float max = Math.Max(a, b);
        float ea = MathF.Exp(a - max);
        float eb = MathF.Exp(b - max);
        float sum = ea + eb;
        float pa = ea / sum;
        float pb = eb / sum;
        return pb >= pa ? (1, pb) : (0, pa);
    }

    /// <summary>
    /// Aligns and crops a face from the source image based on its bounding box.
    /// </summary>
    private static Mat AlignFace(Mat frameBgr, BoundingBox box, int outputSize)
    {
        double w = Math.Max(1, box.Width);
        double h = Math.Max(1, box.Height);
        double cx = box.X + (w / 2.0);
        double cy = box.Y + (h / 2.0);

        // Match upstream "crop_scale = 1.5"
        double maxSide = Math.Max(w, h) * 1.5;
        double scale = outputSize / maxSide;

        // Affine transform: scale around center, map to output center
        double tx = (outputSize / 2.0) - (cx * scale);
        double ty = (outputSize / 2.0) - (cy * scale);
        using Mat m = new(2, 3, MatType.CV_64FC1);
        m.Set(0, 0, scale); m.Set(0, 1, 0.0); m.Set(0, 2, tx);
        m.Set(1, 0, 0.0); m.Set(1, 1, scale); m.Set(1, 2, ty);

        Mat aligned = new();
        Cv2.WarpAffine(frameBgr, aligned, m, new Size(outputSize, outputSize), InterpolationFlags.Linear,
            BorderTypes.Constant, Scalar.Black);
        return aligned;
    }

    /// <summary>
    /// Introspects model metadata to identify the correct image input node.
    /// </summary>
    private static (string inputName, int inputSize, bool nhwc) SelectImageInput(IReadOnlyDictionary<string, NodeMetadata> inputs)
    {
        foreach (KeyValuePair<string, NodeMetadata> kvp in inputs.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            NodeMetadata meta = kvp.Value;
            if (meta.ElementType != typeof(float))
            {
                continue;
            }

            int[] dims = meta.Dimensions;
            if (dims.Length != 4)
            {
                continue;
            }

            bool nhwc = dims[3] == 3;
            int size = nhwc ? dims[1] : dims[2];
            if (size <= 0)
            {
                size = 96;
            }

            return (kvp.Key, size, nhwc);
        }

        if (inputs.Count == 1)
        {
            KeyValuePair<string, NodeMetadata> only = inputs.First();
            int[] dims = only.Value.Dimensions;
            int size = (dims.Length == 4 && dims[2] > 0) ? dims[2] : 96;
            bool nhwc = dims.Length == 4 && dims[3] == 3;
            return (only.Key, size, nhwc);
        }

        throw new InvalidOperationException("GenderAge model input not recognized. Expected a float image tensor input.");
    }

    /// <summary>
    /// Selects the primary output node from the model metadata.
    /// </summary>
    private static string SelectOutput(IReadOnlyDictionary<string, NodeMetadata> outputs)
    {
        foreach (KeyValuePair<string, NodeMetadata> kvp in outputs.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            NodeMetadata meta = kvp.Value;
            if (meta.ElementType != typeof(float))
            {
                continue;
            }

            int[] dims = meta.Dimensions;
            if (dims.Length == 2 && (dims[1] == 3 || dims[1] <= 0))
            {
                return kvp.Key;
            }
        }

        return outputs.Keys.First();
    }
}
