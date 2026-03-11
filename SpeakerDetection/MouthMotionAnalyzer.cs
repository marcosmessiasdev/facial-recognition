using OpenCvSharp;

namespace SpeakerDetection;

/// <summary>
/// Per-track motion analyzer that identifies pixel changes in the mouth region.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Quantifies localized facial movement to distinguish the active speaker from passive observers.
///
/// Responsibilities:
/// - Extract a heuristic Region of Interest (ROI) corresponding to the lower face/mouth area.
/// - Compute the absolute difference between consecutive frames of the mouth region.
/// - Maintain a historical window of motion scores per person.
/// - Smooth motion data via average pooling over the temporal window.
///
/// Dependencies:
/// - OpenCvSharp (Image processing and ROI manipulation)
///
/// Architectural Role:
/// Infrastructure Component / Analysis Service.
///
/// Constraints:
/// - Relies on fixed geometric ratios; performance degrades significantly if face crops are 
///   not tightly bounded around the head.
/// </remarks>
/// <remarks>
/// Initializes a new instance of the MouthMotionAnalyzer.
/// </remarks>
/// <param name="bufferMs">Approximate time window (milliseconds) to smooth over.</param>
/// <param name="workWidth">Internal width for motion processing.</param>
/// <param name="workHeight">Internal height for motion processing.</param>
public sealed class MouthMotionAnalyzer(int bufferMs = 1500, int workWidth = 64, int workHeight = 32) : IDisposable
{
    private sealed class TrackState
    {
        public Mat? Prev;
        public Queue<MouthSample> Buffer = new();
        public float? PrevOpenRatio;
        public DateTime PrevOpenRatioUtc;
    }

    private readonly Dictionary<int, TrackState> _states = new();
    private readonly TimeSpan _bufferDuration = TimeSpan.FromMilliseconds(Math.Clamp(bufferMs, 200, 5000));
    private readonly Size _workSize = new(Math.Max(16, workWidth), Math.Max(8, workHeight));

    /// <summary>
    /// Updates the motion state for a specific person.
    /// </summary>
    /// <param name="trackId">The unique ID of the person being tracked.</param>
    /// <param name="faceCropBgrOrBgra">The recent face crop for analysis.</param>
    /// <param name="mouthRoi">
    /// Optional externally calculated mouth ROI. If null, a heuristic-based ROI 
    /// will be calculated from the face crop dimensions.
    /// </param>
    /// <param name="mouthOpenRatio">Optional mouth open ratio derived from landmarks.</param>
    /// <param name="nowUtc">Timestamp for temporal buffering.</param>
    /// <returns>A normalized activity score (higher means more mouth-region activity).</returns>
    public float Update(int trackId, Mat faceCropBgrOrBgra, Rect? mouthRoi, float? mouthOpenRatio, DateTime nowUtc)
    {
        ArgumentNullException.ThrowIfNull(faceCropBgrOrBgra);

        if (faceCropBgrOrBgra.Empty())
        {
            return 0f;
        }

        Rect roi = mouthRoi ?? GetMouthRoi(faceCropBgrOrBgra);
        if (roi.Width < 8 || roi.Height < 8)
        {
            return 0f;
        }

        using Mat mouth = new(faceCropBgrOrBgra, roi);
        using Mat gray = new();
        if (mouth.Channels() == 4)
        {
            Cv2.CvtColor(mouth, gray, ColorConversionCodes.BGRA2GRAY);
        }
        else
        {
            Cv2.CvtColor(mouth, gray, ColorConversionCodes.BGR2GRAY);
        }

        using Mat resized = new();
        Cv2.Resize(gray, resized, _workSize);

        if (!_states.TryGetValue(trackId, out TrackState? state))
        {
            state = new TrackState();
            _states[trackId] = state;
        }

        float diffScore = 0f;
        if (state.Prev != null && !state.Prev.Empty())
        {
            using Mat diff = new();
            Cv2.Absdiff(resized, state.Prev, diff);
            diffScore = (float)(Cv2.Mean(diff).Val0 / 255.0);
        }

        state.Prev?.Dispose();
        state.Prev = resized.Clone();

        float openVel = 0f;
        if (mouthOpenRatio.HasValue)
        {
            if (state.PrevOpenRatio.HasValue)
            {
                double dt = (nowUtc - state.PrevOpenRatioUtc).TotalSeconds;
                if (dt > 1e-3)
                {
                    openVel = (float)(Math.Abs(mouthOpenRatio.Value - state.PrevOpenRatio.Value) / dt);
                }
            }

            state.PrevOpenRatio = mouthOpenRatio.Value;
            state.PrevOpenRatioUtc = nowUtc;
        }

        state.Buffer.Enqueue(new MouthSample(nowUtc, diffScore, openVel));
        while (state.Buffer.Count > 0 && (nowUtc - state.Buffer.Peek().Utc) > _bufferDuration)
        {
            _ = state.Buffer.Dequeue();
        }

        if (state.Buffer.Count == 0)
        {
            return 0f;
        }

        float avgMotion = 0f;
        float avgOpenVel = 0f;
        foreach (MouthSample s in state.Buffer)
        {
            avgMotion += s.Motion;
            avgOpenVel += s.OpenVel;
        }
        avgMotion /= state.Buffer.Count;
        avgOpenVel /= state.Buffer.Count;

        // Normalize open velocity into a 0..1-ish band (empirical; depends on landmarks noise).
        float openVelNorm = Math.Clamp(avgOpenVel / 0.35f, 0f, 1f);

        return (0.70f * avgMotion) + (0.30f * openVelNorm);
    }

    /// <summary>
    /// Updates the analyzer using only motion (no landmarks available).
    /// </summary>
    public float Update(int trackId, Mat faceCropBgrOrBgra, Rect? mouthRoi = null)
        => Update(trackId, faceCropBgrOrBgra, mouthRoi, null, DateTime.UtcNow);


    /// <summary>
    /// Removes internal state for tracks that are no longer active to reclaim memory.
    /// </summary>
    /// <param name="activeTrackIds">The IDs currently being tracked by the system.</param>
    public void PruneToActiveTracks(IEnumerable<int> activeTrackIds)
    {
        HashSet<int> keep = [.. activeTrackIds];
        int[] toRemove = _states.Keys.Where(id => !keep.Contains(id)).ToArray();
        foreach (int id in toRemove)
        {
            if (_states.TryGetValue(id, out TrackState? st))
            {
                st.Prev?.Dispose();
            }
            _ = _states.Remove(id);
        }
    }

    /// <summary>
    /// Estimates the mouth area coordinates within a face crop.
    /// </summary>
    private static Rect GetMouthRoi(Mat face)
    {
        int w = face.Width;
        int h = face.Height;

        // Approximate mouth region: lower-middle portion of the face crop.
        int x = (int)(w * 0.20);
        int y = (int)(h * 0.60);
        int rw = (int)(w * 0.60);
        int rh = (int)(h * 0.30);

        x = Math.Clamp(x, 0, Math.Max(0, w - 1));
        y = Math.Clamp(y, 0, Math.Max(0, h - 1));
        rw = Math.Clamp(rw, 1, w - x);
        rh = Math.Clamp(rh, 1, h - y);

        return new Rect(x, y, rw, rh);
    }

    /// <summary>
    /// Releases unmanaged OpenCV Mat resources.
    /// </summary>
    public void Dispose()
    {
        foreach (TrackState st in _states.Values)
        {
            st.Prev?.Dispose();
        }

        _states.Clear();
    }

    private readonly record struct MouthSample(DateTime Utc, float Motion, float OpenVel);
}
