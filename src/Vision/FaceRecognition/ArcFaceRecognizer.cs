using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceRecognition;

/// <summary>
/// Component used to extract unique facial feature embeddings using the ArcFace deep learning architecture.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Transforms a cropped face image into a 512-dimensional mathematical vector (embedding) 
/// that represents facial features for identity verification.
///
/// Responsibilities:
/// - Manage the ONNX Runtime inference session for the ArcFace model.
/// - Resize and normalize input face crops to the expected model format (112x112).
/// - Execute model inference to generate raw feature vectors.
/// - Apply L2-normalization to the resulting embeddings for consistent distance calculations.
///
/// Dependencies:
/// - Microsoft.ML.OnnxRuntime (Model execution)
/// - OpenCvSharp (Image preprocessing and resizing)
///
/// Architectural Role:
/// Infrastructure Component / Recognition Service. It provides the "identity signature" used by the IdentityStore.
///
/// Constraints:
/// - Input Mat must contain a face crop (ideally centered).
/// - The model must be a valid ArcFace ONNX model with compatible input/output layers.
/// </remarks>
public class ArcFaceRecognizer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;

    private const int FaceSize = 112;

    /// <summary>
    /// Initializes a new instance of the ArcFaceRecognizer class.
    /// </summary>
    /// <param name="modelPath">
    /// The filesystem path to the pre-trained ArcFace ONNX model.
    /// </param>
    public ArcFaceRecognizer(string modelPath)
    {
        using SessionOptions options = new();
        _session = new InferenceSession(modelPath, options);
        _inputName = SelectImageInputName(_session.InputMetadata);
        _outputName = SelectEmbeddingOutputName(_session.OutputMetadata);
    }

    /// <summary>
    /// Generates a normalized facial embedding from a cropped image of a face.
    /// </summary>
    /// <param name="faceCrop">
    /// An OpenCV Mat containing the cropped facial image. 
    /// It will be automatically resized to 112x112 internally.
    /// </param>
    /// <returns>
    /// A 512-dimensional float array representing the facial identity.
    /// </returns>
    /// <remarks>
    /// This method performs preprocessing (normalization: (pixel - 127.5) / 128.0) 
    /// before running the inference session and L2-normalizing the output.
    /// </remarks>
    public float[] GetEmbedding(Mat faceCrop)
    {
        ArgumentNullException.ThrowIfNull(faceCrop);

        using Mat resized = new();
        Cv2.Resize(faceCrop, resized, new Size(FaceSize, FaceSize));

        using Mat rgb = new();
        if (resized.Channels() == 4)
        {
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGRA2RGB);
        }
        else
        {
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
        }

        int[] inputDims = _session.InputMetadata[_inputName].Dimensions;
        bool nhwc = IsNhwc(inputDims);

        DenseTensor<float> tensor = nhwc
            ? new DenseTensor<float>([1, FaceSize, FaceSize, 3])
            : new DenseTensor<float>([1, 3, FaceSize, FaceSize]);

        unsafe
        {
            byte* ptr = rgb.DataPointer;
            int stride = (int)rgb.Step();
            for (int y = 0; y < FaceSize; y++)
            {
                for (int x = 0; x < FaceSize; x++)
                {
                    int idx = (y * stride) + (x * 3);
                    // ArcFace preprocess: (pixel - 127.5) / 128.0
                    float r = (ptr[idx + 0] - 127.5f) / 128.0f;
                    float g = (ptr[idx + 1] - 127.5f) / 128.0f;
                    float b = (ptr[idx + 2] - 127.5f) / 128.0f;

                    if (nhwc)
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

        // L2-normalize the embedding vector
        return L2Normalize(output);
    }

    /// <summary>
    /// Performs L2 normalization on a vector to ensure it has a unit length.
    /// </summary>
    /// <param name="v">The raw feature vector from the model.</param>
    /// <returns>The normalized vector.</returns>
    private static float[] L2Normalize(float[] v)
    {
        float norm = 0f;
        foreach (float x in v)
        {
            norm += x * x;
        }

        norm = MathF.Sqrt(norm);
        if (norm < 1e-10f)
        {
            return v;
        }

        float[] result = new float[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = v[i] / norm;
        }

        return result;
    }

    /// <summary>
    /// Releases the ONNX Runtime inference session resources.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Determines if the model input tensor is in NHWC (Channels Last) format.
    /// </summary>
    private static bool IsNhwc(int[] dims)
    {
        return dims.Length == 4 && dims[3] == 3;
    }

    /// <summary>
    /// Validates if the input tensor dimensions match expected image patterns (112x112, 3 channels).
    /// </summary>
    private static bool MatchesImageInput(int[] dims)
    {
        if (dims.Length != 4)
        {
            return false;
        }

        bool has112 = dims.Contains(FaceSize) || dims.Any(d => d <= 0);
        bool nchw = (dims[1] == 3 || dims[1] <= 0) && (dims[2] == FaceSize || dims[2] <= 0) && (dims[3] == FaceSize || dims[3] <= 0);
        bool nhwc = (dims[3] == 3 || dims[3] <= 0) && (dims[1] == FaceSize || dims[1] <= 0) && (dims[2] == FaceSize || dims[2] <= 0);
        return has112 && (nchw || nhwc);
    }

    /// <summary>
    /// Automatically identifies the correct input name for the image tensor in the ONNX model.
    /// </summary>
    private static string SelectImageInputName(IReadOnlyDictionary<string, NodeMetadata> inputs)
    {
        foreach (KeyValuePair<string, NodeMetadata> kvp in inputs)
        {
            NodeMetadata meta = kvp.Value;
            if (meta.ElementType != typeof(float))
            {
                continue;
            }

            if (!MatchesImageInput(meta.Dimensions))
            {
                continue;
            }

            return kvp.Key;
        }

        // Fallback: models with a single input
        return inputs.Count == 1
            ? inputs.Keys.First()
            : throw new InvalidOperationException("ArcFace model input not recognized. Expected a float image tensor input.");
    }

    /// <summary>
    /// Automatically identifies the output name for the facial embedding vector.
    /// </summary>
    private static string SelectEmbeddingOutputName(IReadOnlyDictionary<string, NodeMetadata> outputs)
    {
        foreach (KeyValuePair<string, NodeMetadata> kvp in outputs)
        {
            NodeMetadata meta = kvp.Value;
            if (meta.ElementType != typeof(float))
            {
                continue;
            }

            int[] dims = meta.Dimensions;
            if (dims.Length == 2 && (dims[1] == 512 || dims[1] <= 0))
            {
                return kvp.Key;
            }
        }

        return outputs.Keys.First();
    }
}
