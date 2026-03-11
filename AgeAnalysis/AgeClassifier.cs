using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace AgeAnalysis;

/// <summary>
/// Represents broad age categories for classification.
/// </summary>
public enum AgeBucket
{
    Age0To2,
    Age4To6,
    Age8To12,
    Age15To20,
    Age25To32,
    Age38To43,
    Age48To53,
    Age60To100
}

/// <summary>
/// Classifies the age of a person from their face image using an ONNX model.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Categorizes individuals into age groups to provide demographic insights.
///
/// Responsibilities:
/// - Load and manage an ONNX age classification model.
/// - Preprocess face crops (BGR to RGB, resizing, mean subtraction) to match model requirements.
/// - Execute inference and apply Softmax to obtain probability distribution across age buckets.
/// - Map model output indices to the AgeBucket enum and descriptive labels.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime (Inference Engine)
/// - OpenCvSharp (Image preprocessing)
///
/// Architectural Role:
/// Infrastructure Component / Analysis Service.
///
/// Constraints:
/// - Model expects 224x224 RGB inputs with specific mean subtraction.
/// </remarks>
public sealed class AgeClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;

    private const int FaceSize = 224;

    // Matches ONNX Model Zoo sample preprocessing for age_googlenet.onnx
    private static readonly float[] MeanRgb = [104f, 117f, 123f];

    private static readonly AgeBucket[] Labels =
    [
        AgeBucket.Age0To2,
        AgeBucket.Age4To6,
        AgeBucket.Age8To12,
        AgeBucket.Age15To20,
        AgeBucket.Age25To32,
        AgeBucket.Age38To43,
        AgeBucket.Age48To53,
        AgeBucket.Age60To100
    ];

    private static readonly string[] LabelText =
    [
        "(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"
    ];

    /// <summary>
    /// Initializes a new instance of the AgeClassifier with the specified model.
    /// </summary>
    /// <param name="modelPath">The absolute path to the .onnx age model file.</param>
    public AgeClassifier(string modelPath)
    {
        using SessionOptions options = new();
        _session = new InferenceSession(modelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Categorizes the age of a given face crop.
    /// </summary>
    /// <param name="faceCrop">A BGR or BGRA image containing the face region.</param>
    /// <returns>A tuple containing the AgeBucket, a display label, and the confidence score.</returns>
    public (AgeBucket bucket, string label, float confidence) Classify(Mat faceCrop)
    {
        ArgumentNullException.ThrowIfNull(faceCrop);

        using Mat bgr = new();
        switch (faceCrop.Channels())
        {
            case 4:
                Cv2.CvtColor(faceCrop, bgr, ColorConversionCodes.BGRA2BGR);
                break;
            case 3:
                faceCrop.CopyTo(bgr);
                break;
            default:
                Cv2.CvtColor(faceCrop, bgr, ColorConversionCodes.GRAY2BGR);
                break;
        }

        using Mat rgb = new();
        Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

        using Mat resized = new();
        Cv2.Resize(rgb, resized, new Size(FaceSize, FaceSize));

        DenseTensor<float> tensor = new([1, 3, FaceSize, FaceSize]);

        unsafe
        {
            byte* ptr = resized.DataPointer;
            int stride = (int)resized.Step();

            for (int y = 0; y < FaceSize; y++)
            {
                for (int x = 0; x < FaceSize; x++)
                {
                    int offset = (y * stride) + (x * 3);
                    float r = ptr[offset + 0] - MeanRgb[0];
                    float g = ptr[offset + 1] - MeanRgb[1];
                    float b = ptr[offset + 2] - MeanRgb[2];

                    tensor[0, 0, y, x] = r;
                    tensor[0, 1, y, x] = g;
                    tensor[0, 2, y, x] = b;
                }
            }
        }

        List<NamedOnnxValue> inputs = new()
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        float[] scores = results.First(r => r.Name == _outputName).AsEnumerable<float>().ToArray();

        float maxScore = scores.Max();
        float sumExp = scores.Sum(s => MathF.Exp(s - maxScore));
        float[] probs = scores.Select(s => MathF.Exp(s - maxScore) / sumExp).ToArray();

        int bestIdx = Array.IndexOf(probs, probs.Max());
        AgeBucket bucket = bestIdx < Labels.Length ? Labels[bestIdx] : AgeBucket.Age25To32;
        string label = bestIdx < LabelText.Length ? LabelText[bestIdx] : "(?)";
        return (bucket, label, probs[bestIdx]);
    }

    /// <summary>
    /// Releases the ONNX runtime session resources.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}

