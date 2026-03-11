using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace EmotionAnalysis
{
    /// <summary>
    /// Defines the supported facial expression categories.
    /// </summary>
    public enum Emotion
    {
        Neutral, Happy, Surprise, Sad, Angry, Disgust, Fear, Contempt
    }

    /// <summary>
    /// Classifies facial expressions into emotion categories using an ONNX model.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Analyzes the emotional state of a tracked person by classifying their facial expression.
    ///
    /// Responsibilities:
    /// - Load and manage an ONNX emotion classification model.
    /// - Preprocess face crops (grayscale conversion, resizing, normalization) to match model requirements.
    /// - Execute inference and apply Softmax to scores to obtain probability distribution.
    /// - Map model output indices to the Emotion enum.
    ///
    /// Dependencies:
    /// - Microsoft.ML.OnnxRuntime (Inference Engine)
    /// - OpenCvSharp (Image preprocessing)
    ///
    /// Architectural Role:
    /// Infrastructure Component / Analysis Service.
    ///
    /// Constraints:
    /// - Model typically expects 64x64 grayscale inputs.
    /// </remarks>
    public class EmotionClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _faceSize;

        private static readonly Emotion[] Labels =
        {
            Emotion.Neutral, Emotion.Happy, Emotion.Surprise, Emotion.Sad,
            Emotion.Angry, Emotion.Disgust, Emotion.Fear, Emotion.Contempt
        };

        /// <summary>
        /// Initializes a new instance of the EmotionClassifier with the specified model.
        /// </summary>
        /// <param name="modelPath">The absolute path to the .onnx emotion model file.</param>
        public EmotionClassifier(string modelPath)
        {
            _session = new InferenceSession(modelPath, new SessionOptions());
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            var dims = _session.InputMetadata[_inputName].Dimensions;
            _faceSize = dims.Length >= 4 && dims[2] > 0 ? dims[2] : 64;
        }

        /// <summary>
        /// Classifies the emotion of a given face crop.
        /// </summary>
        /// <param name="faceCrop">A BGR or BGRA image containing the face region.</param>
        /// <returns>A tuple containing the predicted Emotion and the confidence score (0.0 to 1.0).</returns>
        public (Emotion emotion, float confidence) Classify(Mat faceCrop)
        {
            using var gray = new Mat();
            switch (faceCrop.Channels())
            {
                case 4:
                    Cv2.CvtColor(faceCrop, gray, ColorConversionCodes.BGRA2GRAY);
                    break;
                case 3:
                    Cv2.CvtColor(faceCrop, gray, ColorConversionCodes.BGR2GRAY);
                    break;
                default:
                    faceCrop.CopyTo(gray);
                    break;
            }

            using var resized = new Mat();
            Cv2.Resize(gray, resized, new Size(_faceSize, _faceSize));

            var tensor = new DenseTensor<float>(new[] { 1, 1, _faceSize, _faceSize });

            unsafe
            {
                byte* ptr = (byte*)resized.DataPointer;
                for (int y = 0; y < _faceSize; y++)
                    for (int x = 0; x < _faceSize; x++)
                        tensor[0, 0, y, x] = ptr[y * _faceSize + x] / 255.0f;
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);
            var scores = results.First(r => r.Name == _outputName).AsEnumerable<float>().ToArray();

            // Softmax to get probabilities  
            float maxScore = scores.Max();
            float sumExp = scores.Sum(s => MathF.Exp(s - maxScore));
            var probs = scores.Select(s => MathF.Exp(s - maxScore) / sumExp).ToArray();

            int bestIdx = Array.IndexOf(probs, probs.Max());
            var label = bestIdx < Labels.Length ? Labels[bestIdx] : Emotion.Neutral;
            return (label, probs[bestIdx]);
        }

        /// <summary>
        /// Releases the ONNX runtime session resources.
        /// </summary>
        public void Dispose() => _session?.Dispose();
    }
}

