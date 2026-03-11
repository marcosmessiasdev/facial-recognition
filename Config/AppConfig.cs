using Microsoft.Extensions.Configuration;
using System.IO;
using System.Globalization;
using System.Reflection;
using System;

namespace Config
{
    /// <summary>
    /// Centralized configuration class for the facial recognition system.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Provides a single point of truth for application settings, performance constants, and model paths.
    ///
    /// Responsibilities:
    /// - Define defaults for all system parameters (FPS, intervals, thresholds).
    /// - Load and bind settings from external providers (appsettings.json).
    /// - Handle snake_case to PascalCase mapping for consistent configuration naming.
    ///
    /// Dependencies:
    /// - Microsoft.Extensions.Configuration (Setting management)
    ///
    /// Architectural Role:
    /// Infrastructure Component / Configuration Provider.
    ///
    /// Constraints:
    /// - Properties must be public and writable for typical binding to work.
    /// </remarks>
    public class AppConfig
    {
        /// <summary>Target capture frames per second.</summary>
        public int CaptureFps { get; set; } = 30;

        /// <summary>Interval (in frames) between full face detection passes.</summary>
        public int FaceDetectionInterval { get; set; } = 5;

        /// <summary>Interval (in frames) between emotion analysis updates.</summary>
        public int EmotionInterval { get; set; } = 10;

        /// <summary>Interval (in frames) for simultaneous age/gender attribute updates.</summary>
        public int AttributesInterval { get; set; } = 15;

        /// <summary>Interval (in frames) for dedicated gender classification updates.</summary>
        public int GenderInterval { get; set; } = 15;

        /// <summary>Interval (in frames) for dedicated age classification updates.</summary>
        public int AgeInterval { get; set; } = 20;

        /// <summary>Enables or disables gender prediction.</summary>
        public bool EnableGenderPrediction { get; set; } = true;

        /// <summary>Enables or disables age prediction.</summary>
        public bool EnableAgePrediction { get; set; } = true;

        /// <summary>Enables or disables Voice Activity Detection for presence verification.</summary>
        public bool EnableAudioVad { get; set; } = true;

        /// <summary>
        /// Audio source to use for VAD/transcription: "loopback" (system output) or "microphone".
        /// </summary>
        public string AudioSource { get; set; } = "loopback";

        /// <summary>Minimum probability threshold for VAD to consider a signal as speech.</summary>
        public float AudioVadSpeechThreshold { get; set; } = 0.6f;

        /// <summary>Frame size for audio analysis buffers.</summary>
        public int AudioVadFrameSizeSamples { get; set; } = 512;

        /// <summary>Sample rate for incoming loopback audio.</summary>
        public int AudioVadSampleRateHz { get; set; } = 16000;

        /// <summary>
        /// Optional output device selector for loopback capture (matches device ID or a substring of FriendlyName).
        /// If null/empty, the default render endpoint is used.
        /// </summary>
        public string? AudioLoopbackDevice { get; set; } = null;

        /// <summary>
        /// Optional input device selector for microphone capture (matches device ID or a substring of FriendlyName).
        /// If null/empty, the default capture endpoint is used.
        /// </summary>
        public string? AudioMicrophoneDevice { get; set; } = null;

        /// <summary>
        /// If true, highlights the most likely speaker using visual mouth motion even when audio VAD is inactive.
        /// Useful when loopback capture is muted or unavailable.
        /// </summary>
        public bool EnableVisualSpeakerFallback { get; set; } = true;

        /// <summary>Enables or disables dense FaceMesh landmarks for improved mouth features.</summary>
        public bool EnableFaceMeshLandmarks { get; set; } = true;

        /// <summary>Interval (in frames) between FaceMesh landmark updates per tracked face.</summary>
        public int FaceMeshIntervalFrames { get; set; } = 3;

        /// <summary>Similarity threshold for face recognition (lower is stricter).</summary>
        public double RecognitionThreshold { get; set; } = 0.40;

        /// <summary>Intersection over Union threshold for face tracking association.</summary>
        public float IouThreshold { get; set; } = 0.3f;

        /// <summary>Number of consecutive frames a face can be absent before tracking is lost.</summary>
        public int MaxMissedFrames { get; set; } = 5;

        /// <summary>Interval (in frames) for re-verifying a tracked person's identity.</summary>
        public int RecognitionIntervalFrames { get; set; } = 15;

        /// <summary>Relative path to the face detection (SCRFD) model file.</summary>
        public string ModelScrfd { get; set; } = "onnx/scrfd_2.5g_kps.onnx";

        /// <summary>Relative path to the face recognition (ArcFace) model file.</summary>
        public string ModelArcface { get; set; } = "onnx/arcface.onnx";

        /// <summary>Relative path to the emotion classification model file.</summary>
        public string ModelFer2013 { get; set; } = "onnx/emotion-ferplus-8.onnx";

        /// <summary>Relative path to the combined gender/age classification model file.</summary>
        public string ModelGenderAge { get; set; } = "onnx/genderage.onnx";

        /// <summary>Relative path to the dedicated gender classification model file.</summary>
        public string ModelGender { get; set; } = "onnx/gender_googlenet.onnx";

        /// <summary>Relative path to the dedicated age classification model file.</summary>
        public string ModelAge { get; set; } = "onnx/age_googlenet.onnx";

        /// <summary>Relative path to the Silero VAD model file.</summary>
        public string ModelSileroVad { get; set; } = "onnx/silero_vad.onnx";

        /// <summary>Relative path to the FaceMesh landmarks model file.</summary>
        public string ModelFaceMesh { get; set; } = "onnx/face_mesh_Nx3x192x192_post.onnx";

        /// <summary>Enables or disables offline speech-to-text transcription (Whisper).</summary>
        public bool EnableTranscription { get; set; } = true;

        /// <summary>Language hint for transcription (e.g. "pt", "en", or "auto").</summary>
        public string TranscriptionLanguage { get; set; } = "pt";

        /// <summary>Max seconds per transcription segment.</summary>
        public int TranscriptionMaxSegmentSeconds { get; set; } = 12;

        /// <summary>Silence hangover before closing a segment.</summary>
        public int TranscriptionHangoverMs { get; set; } = 350;

        /// <summary>Relative path to the Whisper GGML model file.</summary>
        public string ModelWhisperGgml { get; set; } = "models/whisper/ggml-tiny.bin";

        /// <summary>
        /// Loads configuration from appsettings.json relative to the application base directory.
        /// Falls back to defaults if file is not found.
        /// </summary>
        /// <returns>An initialized AppConfig instance.</returns>
        public static AppConfig Load()
        {
            var basedir = System.AppDomain.CurrentDomain.BaseDirectory;
            var jsonPath = Path.Combine(basedir, "appsettings.json");

            var config = new ConfigurationBuilder()
                .SetBasePath(basedir)
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
                .Build();

            var appConfig = new AppConfig();
            config.Bind(appConfig);
            ApplySnakeCaseOverrides(config, appConfig);
            return appConfig;
        }

        /// <summary>
        /// Iterates over configuration keys to find snake_case versions of property names 
        /// and applies them to the config object.
        /// </summary>
        private static void ApplySnakeCaseOverrides(IConfiguration config, AppConfig cfg)
        {
            foreach (var prop in typeof(AppConfig).GetProperties(BindingFlags.Instance | BindingFlags.Public))
            {
                if (!prop.CanWrite) continue;

                var key = ToSnakeCase(prop.Name);
                var raw = config[key];
                if (string.IsNullOrWhiteSpace(raw)) continue;

                if (TryConvert(raw, prop.PropertyType, out var value))
                {
                    prop.SetValue(cfg, value);
                }
            }
        }

        /// <summary>
        /// Attempts to convert a raw string value from config into the target property type.
        /// </summary>
        private static bool TryConvert(string raw, Type type, out object? value)
        {
            value = null;

            if (type == typeof(string))
            {
                value = raw;
                return true;
            }

            if (type == typeof(bool))
            {
                if (bool.TryParse(raw, out var b)) { value = b; return true; }
                return false;
            }

            if (type == typeof(int))
            {
                if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i)) { value = i; return true; }
                return false;
            }

            if (type == typeof(float))
            {
                if (float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var f)) { value = f; return true; }
                return false;
            }

            if (type == typeof(double))
            {
                if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) { value = d; return true; }
                return false;
            }

            return false;
        }

        /// <summary>
        /// Convers a PascalCase string to snake_case.
        /// </summary>
        private static string ToSnakeCase(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;

            var result = new System.Text.StringBuilder(name.Length + 8);
            for (int i = 0; i < name.Length; i++)
            {
                var c = name[i];
                if (char.IsUpper(c))
                {
                    if (i > 0) result.Append('_');
                    result.Append(char.ToLowerInvariant(c));
                }
                else
                {
                    result.Append(c);
                }
            }

            return result.ToString();
        }
    }
}
