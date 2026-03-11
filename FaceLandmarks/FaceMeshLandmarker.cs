using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceLandmarks;

/// <summary>
/// Computes dense face landmarks using a MediaPipe FaceMesh ONNX model and derives mouth metrics.
/// </summary>
public sealed class FaceMeshLandmarker : IDisposable
{
    private const int InputSize = 192;

    // MediaPipe FaceMesh indices commonly used for mouth metrics.
    private const int MouthUpperInner = 13;
    private const int MouthLowerInner = 14;
    private const int MouthLeftCorner = 78;
    private const int MouthRightCorner = 308;

    private readonly InferenceSession _session;
    private readonly string _imageInputName;
    private readonly (string Name, Type ElementType, int[] Dims)[] _extraInputs;
    private readonly string _landmarksOutputName;

    public FaceMeshLandmarker(string modelPath)
    {
        using SessionOptions opts = new();
        _session = new InferenceSession(modelPath, opts);

        _imageInputName = _session.InputMetadata
            .First(kvp => kvp.Value.Dimensions.Length == 4)
            .Key;

        _extraInputs = [.. _session.InputMetadata
            .Where(kvp => kvp.Key != _imageInputName)
            .Select(kvp => (kvp.Key, kvp.Value.ElementType, kvp.Value.Dimensions))];

        _landmarksOutputName = _session.OutputMetadata.Keys.FirstOrDefault(k => k.Contains("landmark", StringComparison.OrdinalIgnoreCase))
            ?? _session.OutputMetadata.Keys.First();
    }

    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Computes mouth open ratio and a mouth ROI inside the given face crop.
    /// </summary>
    public bool TryGetMouthMetrics(Mat faceCropBgrOrBgra, out float mouthOpenRatio, out Rect mouthRoi)
    {
        mouthOpenRatio = 0f;
        mouthRoi = default;

        ArgumentNullException.ThrowIfNull(faceCropBgrOrBgra);

        if (faceCropBgrOrBgra.Empty())
        {
            return false;
        }

        using Mat bgr = EnsureBgr(faceCropBgrOrBgra);
        using Mat resized = new();
        Cv2.Resize(bgr, resized, new Size(InputSize, InputSize));

        DenseTensor<float> input = BuildInputTensorRgb01(resized);

        List<NamedOnnxValue> inputs = new(_extraInputs.Length + 1)
        {
            NamedOnnxValue.CreateFromTensor(_imageInputName, input)
        };

        // Best-effort support for "postprocess" models that require crop parameters.
        foreach ((string? name, Type? elementType, int[]? dims) in _extraInputs)
        {
            inputs.Add(ScalarTensorFor(name, elementType, dims));
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        DisposableNamedOnnxValue? output = null;
        foreach (DisposableNamedOnnxValue r in results)
        {
            output ??= r;
            if (string.Equals(r.Name, _landmarksOutputName, StringComparison.Ordinal))
            {
                output = r;
                break;
            }
        }

        if (output == null)
        {
            return false;
        }

        (Pt pUpper, Pt pLower, Pt pLeft, Pt pRight) = ExtractMouthPoints(output);

        float open = Distance(pUpper, pLower);
        float width = Distance(pLeft, pRight);
        if (width <= 1e-6f)
        {
            return false;
        }

        mouthOpenRatio = open / width;

        // ROI around mouth in resized coords.
        float cx = (pLeft.X + pRight.X) * 0.5f;
        float cy = (pUpper.Y + pLower.Y) * 0.5f;
        float mouthW = MathF.Max(8f, width);
        float roiW = mouthW * 1.8f;
        float roiH = mouthW * 1.2f;

        int rx = (int)MathF.Round(cx - (roiW / 2f));
        int ry = (int)MathF.Round(cy - (roiH / 2f));
        int rw = (int)MathF.Round(roiW);
        int rh = (int)MathF.Round(roiH);

        rx = Math.Clamp(rx, 0, InputSize - 1);
        ry = Math.Clamp(ry, 0, InputSize - 1);
        rw = Math.Clamp(rw, 1, InputSize - rx);
        rh = Math.Clamp(rh, 1, InputSize - ry);

        // Scale ROI back to face crop size.
        float sx = (float)faceCropBgrOrBgra.Width / InputSize;
        float sy = (float)faceCropBgrOrBgra.Height / InputSize;
        mouthRoi = new Rect(
            (int)MathF.Round(rx * sx),
            (int)MathF.Round(ry * sy),
            (int)MathF.Round(rw * sx),
            (int)MathF.Round(rh * sy));

        mouthRoi = ClampRect(mouthRoi, faceCropBgrOrBgra.Width, faceCropBgrOrBgra.Height);
        return mouthRoi.Width >= 8 && mouthRoi.Height >= 8;
    }

    private static Rect ClampRect(Rect r, int w, int h)
    {
        int x = Math.Clamp(r.X, 0, Math.Max(0, w - 1));
        int y = Math.Clamp(r.Y, 0, Math.Max(0, h - 1));
        int rw = Math.Clamp(r.Width, 1, w - x);
        int rh = Math.Clamp(r.Height, 1, h - y);
        return new Rect(x, y, rw, rh);
    }

    private static Mat EnsureBgr(Mat input)
    {
        if (input.Channels() == 3)
        {
            return input;
        }

        Mat bgr = new();
        if (input.Channels() == 4)
        {
            Cv2.CvtColor(input, bgr, ColorConversionCodes.BGRA2BGR);
        }
        else
        {
            Cv2.CvtColor(input, bgr, ColorConversionCodes.GRAY2BGR);
        }

        return bgr;
    }

    private static DenseTensor<float> BuildInputTensorRgb01(Mat bgr192)
    {
        // Convert to RGB and normalize to [0..1]. Shape: [1, 3, 192, 192]
        using Mat rgb = new();
        Cv2.CvtColor(bgr192, rgb, ColorConversionCodes.BGR2RGB);

        DenseTensor<float> tensor = new([1, 3, InputSize, InputSize]);
        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                Vec3b px = rgb.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = px.Item0 / 255f;
                tensor[0, 1, y, x] = px.Item1 / 255f;
                tensor[0, 2, y, x] = px.Item2 / 255f;
            }
        }

        return tensor;
    }

    private static NamedOnnxValue ScalarTensorFor(string name, Type elementType, int[] dims)
    {
        // Many postprocess variants take crop coords; for our per-face crop usage we can pass zeros/size.
        float v =
            name.Contains("x2", StringComparison.OrdinalIgnoreCase) ? InputSize :
            name.Contains("y2", StringComparison.OrdinalIgnoreCase) ? InputSize :
            0f;

        int[] shape = dims.Length == 0 ? [1] : dims.Select(d => d > 0 ? d : 1).ToArray();

        if (elementType == typeof(float))
        {
            DenseTensor<float> t = new(shape);
            t.Buffer.Span[0] = v;
            return NamedOnnxValue.CreateFromTensor(name, t);
        }

        if (elementType == typeof(int))
        {
            DenseTensor<int> t = new(shape);
            t.Buffer.Span[0] = (int)MathF.Round(v);
            return NamedOnnxValue.CreateFromTensor(name, t);
        }

        if (elementType == typeof(long))
        {
            DenseTensor<long> t = new(shape);
            t.Buffer.Span[0] = (long)MathF.Round(v);
            return NamedOnnxValue.CreateFromTensor(name, t);
        }

        DenseTensor<float> tf = new(shape);
        tf.Buffer.Span[0] = v;
        return NamedOnnxValue.CreateFromTensor(name, tf);
    }

    private readonly record struct Pt(float X, float Y);

    private static (Pt upper, Pt lower, Pt left, Pt right) ExtractMouthPoints(DisposableNamedOnnxValue output)
    {
        // Prefer float tensors if available.
        if (output.Value is Tensor<float> tf)
        {
            return (
                new Pt(tf[0, MouthUpperInner, 0], tf[0, MouthUpperInner, 1]),
                new Pt(tf[0, MouthLowerInner, 0], tf[0, MouthLowerInner, 1]),
                new Pt(tf[0, MouthLeftCorner, 0], tf[0, MouthLeftCorner, 1]),
                new Pt(tf[0, MouthRightCorner, 0], tf[0, MouthRightCorner, 1]));
        }

        if (output.Value is Tensor<int> ti)
        {
            return (
                new Pt(ti[0, MouthUpperInner, 0], ti[0, MouthUpperInner, 1]),
                new Pt(ti[0, MouthLowerInner, 0], ti[0, MouthLowerInner, 1]),
                new Pt(ti[0, MouthLeftCorner, 0], ti[0, MouthLeftCorner, 1]),
                new Pt(ti[0, MouthRightCorner, 0], ti[0, MouthRightCorner, 1]));
        }

        if (output.Value is Tensor<long> tl)
        {
            return (
                new Pt(tl[0, MouthUpperInner, 0], tl[0, MouthUpperInner, 1]),
                new Pt(tl[0, MouthLowerInner, 0], tl[0, MouthLowerInner, 1]),
                new Pt(tl[0, MouthLeftCorner, 0], tl[0, MouthLeftCorner, 1]),
                new Pt(tl[0, MouthRightCorner, 0], tl[0, MouthRightCorner, 1]));
        }

        if (output.Value is Tensor<double> td)
        {
            return (
                new Pt((float)td[0, MouthUpperInner, 0], (float)td[0, MouthUpperInner, 1]),
                new Pt((float)td[0, MouthLowerInner, 0], (float)td[0, MouthLowerInner, 1]),
                new Pt((float)td[0, MouthLeftCorner, 0], (float)td[0, MouthLeftCorner, 1]),
                new Pt((float)td[0, MouthRightCorner, 0], (float)td[0, MouthRightCorner, 1]));
        }

        // Fallback: attempt to materialize as float via ToArray (last resort).
        float[] arr = output.AsEnumerable<float>().ToArray();
        int stride = 3;
        int idxU = MouthUpperInner * stride;
        int idxL = MouthLowerInner * stride;
        int idxLC = MouthLeftCorner * stride;
        int idxRC = MouthRightCorner * stride;
        return (
            new Pt(arr[idxU + 0], arr[idxU + 1]),
            new Pt(arr[idxL + 0], arr[idxL + 1]),
            new Pt(arr[idxLC + 0], arr[idxLC + 1]),
            new Pt(arr[idxRC + 0], arr[idxRC + 1]));
    }

    private static float Distance(Pt a, Pt b)
    {
        float dx = a.X - b.X;
        float dy = a.Y - b.Y;
        return MathF.Sqrt((dx * dx) + (dy * dy));
    }
}
