using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace GenderAnalysis
{
    /// <summary>
    /// Represents the estimated biological gender based on facial appearance.
    /// </summary>
    public enum GenderAppearance
    {
        /// <summary>Represents a male facial appearance.</summary>
        Male,
        /// <summary>Represents a female facial appearance.</summary>
        Female
    }

    /// <summary>
    /// Component used to classify the gender of a person based on a cropped image of their face.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Provides an automated way to estimate gender from facial features using a specialized neural network.
    ///
    /// Responsibilities:
    /// - Manage the ONNX Runtime inference session for gender classification.
    /// - Preprocess face crops (resizing to 224x224 and applying mean subtraction).
    /// - Execute model inference and transform raw scores into probabilities using Softmax.
    /// - Determine the most likely gender label and associated confidence score.
    ///
    /// Dependencies:
    /// - Microsoft.ML.OnnxRuntime (Inference engine)
    /// - OpenCvSharp (Image manipulation)
    ///
    /// Architectural Role:
    /// Infrastructure Component / Analysis Service.
    ///
    /// Constraints:
    /// - Performance depends on the quality of the input face crop (detection accuracy).
    /// </remarks>
    public sealed class GenderClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;

        private const int FaceSize = 224;

        // Matches ONNX Model Zoo sample preprocessing for gender_googlenet.onnx
        private static readonly float[] MeanRgb = { 104f, 117f, 123f };

        private static readonly GenderAppearance[] Labels =
        {
            GenderAppearance.Male,
            GenderAppearance.Female
        };

        /// <summary>
        /// Initializes a new instance of the GenderClassifier class.
        /// </summary>
        /// <param name="modelPath">
        /// The filesystem path to the pre-trained gender classification ONNX model.
        /// </param>
        public GenderClassifier(string modelPath)
        {
            _session = new InferenceSession(modelPath, new SessionOptions());
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();
        }

        /// <summary>
        /// Classifies the gender appearance of a face in a cropped image.
        /// </summary>
        /// <param name="faceCrop">
        /// An OpenCV Mat containing the cropped facial image. 
        /// Should be centered on the face for better accuracy.
        /// </param>
        /// <returns>
        /// A tuple containing the predicted GenderAppearance and the confidence score (0 to 1).
        /// </returns>
        /// <remarks>
        /// This method internalizes the specific preprocessing required by the GoogleNet-based 
        /// gender model, including resizing and channel-wise mean subtraction.
        /// </remarks>
        public (GenderAppearance gender, float confidence) Classify(Mat faceCrop)
        {
            using var bgr = new Mat();
            switch (faceCrop.Channels())
            {
                case 4:
                    Cv2.CvtColor(faceCrop, bgr, ColorConversionCodes.BGRA2BGR);
                    break;
                case 3:
                    faceCrop.CopyTo(bgr);
                    break;
                default:
                    Cv2.CvtColor(faceCrop, bgr, ColorConversionCodes.GRAY2BGR);
                    break;
            }

            using var rgb = new Mat();
            Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(FaceSize, FaceSize));

            var tensor = new DenseTensor<float>(new[] { 1, 3, FaceSize, FaceSize });

            unsafe
            {
                byte* ptr = (byte*)resized.DataPointer;
                int stride = (int)resized.Step();

                for (int y = 0; y < FaceSize; y++)
                {
                    for (int x = 0; x < FaceSize; x++)
                    {
                        int offset = y * stride + x * 3;
                        float r = ptr[offset + 0] - MeanRgb[0];
                        float g = ptr[offset + 1] - MeanRgb[1];
                        float b = ptr[offset + 2] - MeanRgb[2];

                        tensor[0, 0, y, x] = r;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = b;
                    }
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);
            var scores = results.First(r => r.Name == _outputName).AsEnumerable<float>().ToArray();

            float maxScore = scores.Max();
            float sumExp = scores.Sum(s => MathF.Exp(s - maxScore));
            var probs = scores.Select(s => MathF.Exp(s - maxScore) / sumExp).ToArray();

            int bestIdx = Array.IndexOf(probs, probs.Max());
            var label = bestIdx < Labels.Length ? Labels[bestIdx] : GenderAppearance.Male;
            return (label, probs[bestIdx]);
        }

        /// <summary>
        /// Releases the ONNX Runtime inference session resources.
        /// </summary>
        public void Dispose() => _session.Dispose();
    }
}



