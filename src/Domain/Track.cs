namespace FacialRecognition.Domain;

/// <summary>
/// Represents the temporal state and accumulated identity information for a single tracked face.
/// </summary>
public sealed class Track
{
    public int Id { get; set; }
    public BoundingBox Box { get; set; } = new();
    public int Missed { get; set; }

    public string? PersonName { get; set; }
    public string? EmotionLabel { get; set; }
    public string? GenderLabel { get; set; }
    public string? AgeLabel { get; set; }

    public float MouthMotionScore { get; set; }
    public float MouthOpenRatio { get; set; }

    public bool IsSpeaking { get; set; }
    public float SpeakingScore { get; set; }
    public float TalkNetSpeakingProb { get; set; }

    public int FramesSinceRecognition { get; set; } = 999;
    public int FramesSinceEmotion { get; set; } = 999;
    public int FramesSinceEmotionDebugLog { get; set; } = 999;
    public int FramesSinceGender { get; set; } = 999;
    public int FramesSinceAge { get; set; } = 999;
    public int FramesSinceLandmarks { get; set; } = 999;
    public int FramesSinceAsd { get; set; } = 999;
}
