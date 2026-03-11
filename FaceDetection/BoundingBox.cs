using System.Drawing;

namespace FaceDetection;

/// <summary>
/// Represents a rectangular region in an image where a face has been detected, 
/// containing geometric coordinates, confidence metrics, and optional key point data.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Transfers geometrical and fidelity information about detection results between different stages 
/// of the vision pipeline (Detection -> Tracking -> Overlay).
///
/// Responsibilities:
/// - Store the absolute pixel coordinates and dimensions of a detected face.
/// - Store the statistical confidence score provided by the detection model.
/// - Hold facial landmark points (eyes, nose, mouth) for alignment.
///
/// Dependencies:
/// - System.Drawing (PointF support)
///
/// Architectural Role:
/// Data Transfer Object (DTO).
/// </remarks>
public class BoundingBox
{
    /// <summary>
    /// Gets or sets the X-coordinate of the upper-left corner of the bounding box.
    /// Value is in absolute pixels relative to the frame width.
    /// </summary>
    public int X { get; set; }

    /// <summary>
    /// Gets or sets the Y-coordinate of the upper-left corner of the bounding box.
    /// Value is in absolute pixels relative to the frame height.
    /// </summary>
    public int Y { get; set; }

    /// <summary>
    /// Gets or sets the width of the bounding box in pixels.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Gets or sets the height of the bounding box in pixels.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Gets or sets the confidence score of the detection, typically between 0.0 and 1.0.
    /// Higher values indicate greater model certainty.
    /// </summary>
    public float Confidence { get; set; }

    /// <summary>
    /// Optional facial landmarks in absolute image coordinates.
    /// Typically contains 5 points (Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner).
    /// </summary>
    public PointF[]? Landmarks { get; set; }
}

