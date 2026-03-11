using FaceDetection;

namespace FaceTracking;

/// <summary>
/// Component responsible for maintaining temporal consistency of detected faces across multiple frames.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Assigns and maintains stable identifiers (IDs) for faces as they move through the video stream. 
/// This prevents redundant recognition processing on the same face in every frame.
///
/// Responsibilities:
/// - Match new detections from the FaceDetector to existing tracks using Intersection over Union (IoU) metrics.
/// - Initialize new tracks for previously unseen faces.
/// - Track the number of missed frames for each face to handle temporary occlusions.
/// - Clean up stale tracks for faces that have left the field of view.
///
/// Dependencies:
/// - FaceDetection (BoundingBox data structures).
///
/// Architectural Role:
/// Infrastructure Component / Middleware. It sits between FaceDetection and the higher-level VisionEngine.
///
/// Constraints:
/// - Relies on IoU, so it performs best with high-framerate input where face movement between frames is localized.
/// </remarks>
/// <remarks>
/// Initializes a new instance of the FaceTracker class.
/// </remarks>
/// <param name="iouThreshold">
/// The minimum Intersection over Union required to consider a detection as a match for an existing track.
/// </param>
/// <param name="maxMissed">
/// The maximum number of consecutive frames a track can be missing a detection before being deleted.
/// </param>
public class FaceTracker(float iouThreshold = 0.3f, int maxMissed = 5)
{
    private readonly List<Track> _tracks = new();
    private int _nextId = 1;

    private readonly int _maxMissed = maxMissed;
    private readonly float _iouThreshold = iouThreshold;

    /// <summary>
    /// Updates the tracker state with a new set of detections from the current frame.
    /// </summary>
    /// <param name="detections">
    /// A list of bounding boxes representing faces detected in the current frame.
    /// </param>
    /// <returns>
    /// An updated list of all active tracks, including matched, new, and pending tracks.
    /// </returns>
    public List<Track> Update(List<BoundingBox> detections)
    {
        ArgumentNullException.ThrowIfNull(detections);

        // Match detections to existing tracks by IoU
        HashSet<int> matched = new();
        HashSet<int> matchedTracks = new();

        foreach (Track track in _tracks)
        {
            track.Missed++;
        }

        for (int di = 0; di < detections.Count; di++)
        {
            float bestIou = 0;
            int bestTrack = -1;

            for (int ti = 0; ti < _tracks.Count; ti++)
            {
                if (matchedTracks.Contains(ti))
                {
                    continue;
                }

                float iou = ComputeIoU(detections[di], _tracks[ti].Box);
                if (iou > bestIou)
                {
                    bestIou = iou;
                    bestTrack = ti;
                }
            }

            if (bestIou >= _iouThreshold && bestTrack >= 0)
            {
                _tracks[bestTrack].Box = detections[di];
                _tracks[bestTrack].Missed = 0;
                _ = matched.Add(di);
                _ = matchedTracks.Add(bestTrack);
            }
        }

        // Create new tracks for unmatched detections
        for (int di = 0; di < detections.Count; di++)
        {
            if (!matched.Contains(di))
            {
                _tracks.Add(new Track
                {
                    Id = _nextId++,
                    Box = detections[di],
                    Missed = 0
                });
            }
        }

        // Remove stale tracks
        _ = _tracks.RemoveAll(t => t.Missed > _maxMissed);

        return [.. _tracks];
    }

    /// <summary>
    /// Computes the Intersection over Union (IoU) of two bounding boxes.
    /// </summary>
    /// <param name="a">The first bounding box.</param>
    /// <param name="b">The second bounding box.</param>
    /// <returns>A float value between 0 and 1 representing the overlap ratio.</returns>
    private static float ComputeIoU(BoundingBox a, BoundingBox b)
    {
        int x1 = Math.Max(a.X, b.X);
        int y1 = Math.Max(a.Y, b.Y);
        int x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        int y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        if (x2 <= x1 || y2 <= y1)
        {
            return 0f;
        }

        float intersection = (float)(x2 - x1) * (y2 - y1);
        float areaA = (float)a.Width * a.Height;
        float areaB = (float)b.Width * b.Height;
        return intersection / (areaA + areaB - intersection);
    }
}

/// <summary>
/// Represents the temporal state and accumulated identity information for a single tracked face.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Acts as a state container for a face as it persists through time, allowing for 
/// intermittent updates of heavy metadata (like identity from ArcFace) while maintaining a high framerate.
///
/// Responsibilities:
/// - Store the stable unique ID for the track.
/// - Store the most recent geometric bounding box.
/// - Keep track of metadata provided by external classifiers (Name, Gender, Age, etc.).
/// - Track the frequency of heavy processing tasks to allow for intelligent throttling.
///
/// Dependencies:
/// - FaceDetection (BoundingBox).
///
/// Architectural Role:
/// Domain Model / State Object.
/// </remarks>
public class Track
{
    /// <summary>
    /// Gets or sets the unique identifier for this track instance.
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// Gets or sets the current bounding box coordinates in the latest frame.
    /// </summary>
    public BoundingBox Box { get; set; } = new();

    /// <summary>
    /// Gets or sets the number of consecutive frames this track has missed a match.
    /// </summary>
    public int Missed { get; set; }

    /// <summary>
    /// Gets or sets the name/identity of the person associated with this track, computed by recognition.
    /// </summary>
    public string? PersonName { get; set; }

    /// <summary>
    /// Gets or sets the detected emotion label for this track.
    /// </summary>
    public string? EmotionLabel { get; set; }

    /// <summary>
    /// Gets or sets the predicted gender label (e.g., "Male", "Female").
    /// </summary>
    public string? GenderLabel { get; set; }

    /// <summary>
    /// Gets or sets the predicted age group label.
    /// </summary>
    public string? AgeLabel { get; set; }

    /// <summary>
    /// Gets or sets the mouth motion score (0..1). Higher means more mouth-region motion.
    /// </summary>
    public float MouthMotionScore { get; set; }

    /// <summary>
    /// Gets or sets the most recent mouth open ratio (dimensionless). Higher means mouth more open.
    /// </summary>
    public float MouthOpenRatio { get; set; }

    /// <summary>
    /// Gets or sets whether this track is currently considered the active speaker.
    /// </summary>
    public bool IsSpeaking { get; set; }

        /// <summary>
        /// Gets or sets the per-face speaking score (0..1) for the current moment.
        /// </summary>
        public float SpeakingScore { get; set; }

        /// <summary>
        /// Gets or sets the per-face speaking probability predicted by TalkNet ASD (0..1).
        /// </summary>
        public float TalkNetSpeakingProb { get; set; }

    /// <summary>
    /// Gets or sets the number of frames elapsed since the last recognition attempt.
    /// </summary>
    /// <remarks>Used for throttling per-track recognition logic.</remarks>
    public int FramesSinceRecognition { get; set; } = 999;

    /// <summary>
    /// Gets or sets the number of frames elapsed since the last emotion analysis attempt.
    /// </summary>
    public int FramesSinceEmotion { get; set; } = 999;

    /// <summary>
    /// Gets or sets the number of frames elapsed since the last gender analysis attempt.
    /// </summary>
    public int FramesSinceGender { get; set; } = 999;

    /// <summary>
    /// Gets or sets the number of frames elapsed since the last age analysis attempt.
    /// </summary>
    public int FramesSinceAge { get; set; } = 999;

    /// <summary>
    /// Gets or sets the number of frames since the last dense landmarks (FaceMesh) update.
    /// </summary>
    public int FramesSinceLandmarks { get; set; } = 999;

    /// <summary>
    /// Gets or sets the number of frames since the last TalkNet ASD update.
    /// </summary>
    public int FramesSinceAsd { get; set; } = 999;
}
