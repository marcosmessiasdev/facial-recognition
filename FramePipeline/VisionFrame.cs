using OpenCvSharp;

namespace FramePipeline;

/// <summary>
/// Represents a single unit of visual data within the processing pipeline.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Encapsulates raw pixel data captured from a source and provides a standardized 
/// OpenCV Mat representation for downstream processing.
///
/// Responsibilities:
/// - Manage the memory and lifecycle of the underlying OpenCV Mat.
/// - Store metadata about the original frame dimensions.
/// - Provide a consistent format (BGRA) for all processing components.
///
/// Dependencies:
/// - OpenCvSharp (Mat handling)
///
/// Architectural Role:
/// Domain Model / Entity. It is the primary data structure flowing through the VisionEngine.
///
/// Constraints:
/// - Must be disposed manually to release unmanaged OpenCV memory.
/// </remarks>
public class VisionFrame : IDisposable
{
    /// <summary>
    /// Gets the underlying OpenCV Mat containing the image data.
    /// </summary>
    public Mat Mat { get; private set; }

    /// <summary>
    /// Gets the original width of the frame at the time of capture.
    /// </summary>
    public int OriginalWidth { get; }

    /// <summary>
    /// Gets the original height of the frame at the time of capture.
    /// </summary>
    public int OriginalHeight { get; }

    /// <summary>
    /// Initializes a new instance of the VisionFrame class from raw pixel data.
    /// </summary>
    /// <param name="pixelData">The raw buffer containing pixel information.</param>
    /// <param name="width">The width of the captured frame.</param>
    /// <param name="height">The height of the captured frame.</param>
    /// <param name="stride">The number of bytes per row in the pixel data.</param>
    public VisionFrame(byte[] pixelData, int width, int height, int stride)
    {
        OriginalWidth = width;
        OriginalHeight = height;

        // Direct3D11 Capture typically gives us B8G8R8A8 format
        // OpenCV's default structure matches Bgra8
        MatType type = MatType.CV_8UC4; // 8-bit, 4 channels (BGRA)

        // We need to use Mat's FromPixelData static method since the constructor taking an array might be internal/private
        Mat = Mat.FromPixelData(height, width, type, pixelData, stride);
    }

    /// <summary>
    /// Releases the unmanaged resources (OpenCV Mat) used by the VisionFrame.
    /// </summary>
    public void Dispose()
    {
        if (Mat != null && !Mat.IsDisposed)
        {
            Mat.Dispose();
            Mat = null!;
        }

        GC.SuppressFinalize(this);
    }
}

