using AgeAnalysis;
using Config;
using EmotionAnalysis;
using FaceAttributes;
using FaceDetection;
using FaceLandmarks;
using FaceRecognition;
using GenderAnalysis;
using Logging;
using SpeakerDetection;

namespace VisionEngine.Services;

[System.Diagnostics.CodeAnalysis.SuppressMessage("Performance", "CA1812:Avoid uninstantiated internal classes", Justification = "Instantiated via dependency injection.")]
internal sealed class VisionModelProvider : IVisionModelProvider
{
    public FaceDetector FaceDetector { get; }
    public ArcFaceRecognizer? Recognizer { get; }
    public EmotionClassifier? Emotion { get; }
    public GenderAgeClassifier? GenderAge { get; }
    public GenderClassifier? Gender { get; }
    public AgeClassifier? Age { get; }
    public FaceMeshLandmarker? FaceMesh { get; }
    public TalkNetAsdModel? TalkNetAsd { get; }

    public VisionModelProvider(AppConfig cfg)
    {
        string basedir = AppDomain.CurrentDomain.BaseDirectory;

        string scrfdPath = Path.Combine(basedir, cfg.ModelScrfd);
        if (!File.Exists(scrfdPath))
        {
            throw new FileNotFoundException($"SCRFD model not found: {scrfdPath}");
        }

        FaceDetector = new FaceDetector(scrfdPath);
        AppLogger.Instance.Information("FaceDetector loaded: {Path}", scrfdPath);

        string arcfacePath = Path.Combine(basedir, cfg.ModelArcface);
        if (File.Exists(arcfacePath))
        {
            Recognizer = new ArcFaceRecognizer(arcfacePath);
            AppLogger.Instance.Information("ArcFaceRecognizer loaded: {Path}", arcfacePath);
        }
        else
        {
            AppLogger.Instance.Warning("ArcFace model not found at {Path} — recognition disabled", arcfacePath);
        }

        string fer2013Path = Path.Combine(basedir, cfg.ModelFer2013);
        if (File.Exists(fer2013Path))
        {
            Emotion = new EmotionClassifier(fer2013Path);
            AppLogger.Instance.Information("EmotionClassifier loaded: {Path}", fer2013Path);
        }
        else
        {
            AppLogger.Instance.Warning("Emotion model not found at {Path} — emotion analysis disabled", fer2013Path);
        }

        bool wantAttributes = cfg.EnableGenderPrediction || cfg.EnableAgePrediction;

        string genderAgePath = Path.Combine(basedir, cfg.ModelGenderAge);
        if (wantAttributes && File.Exists(genderAgePath))
        {
            GenderAge = new GenderAgeClassifier(genderAgePath);
            AppLogger.Instance.Information("GenderAgeClassifier loaded: {Path}", genderAgePath);
        }
        else
        {
            if (wantAttributes)
            {
                AppLogger.Instance.Warning("GenderAge model not found at {Path} — falling back to separate gender/age models", genderAgePath);
            }

            string genderPath = Path.Combine(basedir, cfg.ModelGender);
            if (cfg.EnableGenderPrediction && File.Exists(genderPath))
            {
                Gender = new GenderClassifier(genderPath);
                AppLogger.Instance.Information("GenderClassifier loaded: {Path}", genderPath);
            }
            else if (cfg.EnableGenderPrediction)
            {
                AppLogger.Instance.Warning("Gender model not found at {Path} — gender prediction disabled", genderPath);
            }

            string agePath = Path.Combine(basedir, cfg.ModelAge);
            if (cfg.EnableAgePrediction && File.Exists(agePath))
            {
                Age = new AgeClassifier(agePath);
                AppLogger.Instance.Information("AgeClassifier loaded: {Path}", agePath);
            }
            else if (cfg.EnableAgePrediction)
            {
                AppLogger.Instance.Warning("Age model not found at {Path} — age prediction disabled", agePath);
            }
        }

        if (cfg.EnableFaceMeshLandmarks)
        {
            string faceMeshPath = Path.Combine(basedir, cfg.ModelFaceMesh);
            if (File.Exists(faceMeshPath))
            {
                FaceMesh = new FaceMeshLandmarker(faceMeshPath);
                AppLogger.Instance.Information("FaceMeshLandmarker loaded: {Path}", faceMeshPath);
            }
            else
            {
                AppLogger.Instance.Warning("FaceMesh model not found at {Path} — using heuristic mouth motion only", faceMeshPath);
            }
        }

        if (cfg.EnableTalkNetAsd)
        {
            string talknetPath = Path.Combine(basedir, cfg.ModelTalkNetAsd);
            if (File.Exists(talknetPath))
            {
                TalkNetAsd = new TalkNetAsdModel(talknetPath);
                AppLogger.Instance.Information("TalkNet ASD enabled: {Path}", talknetPath);
            }
            else
            {
                AppLogger.Instance.Warning("TalkNet ASD model not found at {Path} — falling back to heuristics", talknetPath);
            }
        }
    }

    public void Dispose()
    {
        FaceDetector.Dispose();
        Recognizer?.Dispose();
        Emotion?.Dispose();
        GenderAge?.Dispose();
        Gender?.Dispose();
        Age?.Dispose();
        FaceMesh?.Dispose();
        TalkNetAsd?.Dispose();
        GC.SuppressFinalize(this);
    }
}
