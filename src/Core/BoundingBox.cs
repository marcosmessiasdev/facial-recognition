using System.Drawing;

namespace FacialRecognition.Core;

/// <summary>
/// Represents a rectangular region in an image where a face has been detected,
/// containing geometric coordinates, confidence metrics, and optional key point data.
/// </summary>
public sealed class BoundingBox
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }

    /// <summary>Confidence score typically in [0,1].</summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Optional facial landmarks in absolute image coordinates.
    /// Typically contains 5 points (Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner).
    /// </summary>
    public PointF[]? Landmarks { get; set; }
}

