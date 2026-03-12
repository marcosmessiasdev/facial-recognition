using FaceDetection;
using FaceLandmarks;
using FaceRecognition;
using EmotionAnalysis;
using FaceAttributes;
using GenderAnalysis;
using AgeAnalysis;
using SpeakerDetection;

namespace VisionEngine.Services;

internal interface IVisionModelProvider : IDisposable
{
    FaceDetector FaceDetector { get; }

    ArcFaceRecognizer? Recognizer { get; }
    EmotionClassifier? Emotion { get; }

    GenderAgeClassifier? GenderAge { get; }
    GenderClassifier? Gender { get; }
    AgeClassifier? Age { get; }

    FaceMeshLandmarker? FaceMesh { get; }
    TalkNetAsdModel? TalkNetAsd { get; }
}

