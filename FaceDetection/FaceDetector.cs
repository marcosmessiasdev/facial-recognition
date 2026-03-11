using System;
using System.Collections.Generic;
using FaceAiSharp;
using Microsoft.Extensions.Caching.Memory;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using OpenCvSharp;
using FramePipeline;

namespace FaceDetection
{
    /// <summary>
    /// Component responsible for detecting faces within vision frames using deep learning models.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Provides an abstraction for face detection logic, isolating the specific underlying detection 
    /// engine (FaceAiSharp) from the rest of the application.
    ///
    /// Responsibilities:
    /// - Initialize and manage the SCRFD face detection model.
    /// - Convert frame data from OpenCV formats to ImageSharp formats required by the detector.
    /// - Execute face detection on provided image data.
    /// - Filter detection results based on a confidence threshold.
    ///
    /// Dependencies:
    /// - FaceAiSharp (SCRFD Detector)
    /// - Microsoft.Extensions.Caching.Memory (For model caching)
    /// - SixLabors.ImageSharp (Image processing for detection)
    /// - OpenCvSharp (Frame handling)
    ///
    /// Architectural Role:
    /// Infrastructure Component / Service. It sits between the frame capture and the recognition/analysis stages.
    ///
    /// Constraints:
    /// - Must receive a valid, non-empty VisionFrame.
    /// - Handles detection logic only; no recognition or tracking.
    /// </remarks>
    public class FaceDetector : IDisposable
    {
        private readonly IFaceDetectorWithLandmarks _detector;
        private readonly MemoryCache _cache;
        
        /// <summary>
        /// Initializes a new instance of the FaceDetector class using the SCRFD engine.
        /// </summary>
        /// <param name="modelPath">
        /// The filesystem path to the pre-trained ONNX model file. 
        /// Must be a valid SCRFD model with landmark support.
        /// </param>
        public FaceDetector(string modelPath)
        {
            _cache = new MemoryCache(new MemoryCacheOptions());
            _detector = new FaceAiSharp.ScrfdDetector(_cache, new FaceAiSharp.ScrfdDetectorOptions { ModelPath = modelPath });
        }

        /// <summary>
        /// Analyzes a vision frame to locate human faces and their key landmarks.
        /// </summary>
        /// <param name="frame">
        /// The vision frame containing the BGRA image data to analyze. 
        /// Frame must not be null or disposed.
        /// </param>
        /// <returns>
        /// A list of <see cref="BoundingBox"/> objects representing the detected faces. 
        /// Only faces with a confidence above 0.5 are included.
        /// </returns>
        /// <remarks>
        /// This method executes a multi-step pipeline:
        /// 1. Converts OpenCV BGRA color space to RGB using <see cref="Cv2.CvtColor"/>.
        /// 2. Marshals the raw pointer data into a managed byte array.
        /// 3. Loads the data into a <see cref="SixLabors.ImageSharp.Image"/> for model ingestion.
        /// 4. Maps FaceAiSharp results to the local DTO format, including landmark conversion.
        /// </remarks>
        public List<BoundingBox> Detect(VisionFrame frame)
        {
            var boxes = new List<BoundingBox>();
            if (frame.Mat == null || frame.Mat.IsDisposed || frame.Mat.Empty()) 
                return boxes;

            int width = frame.Mat.Width;
            int height = frame.Mat.Height;
            
            using var rgb = new Mat();
            Cv2.CvtColor(frame.Mat, rgb, ColorConversionCodes.BGRA2RGB);
            
            byte[] rgbData = new byte[width * height * 3];
            unsafe 
            {
                System.Runtime.InteropServices.Marshal.Copy((IntPtr)rgb.DataPointer, rgbData, 0, rgbData.Length);
            }
            
            using var image = SixLabors.ImageSharp.Image.LoadPixelData<Rgb24>(rgbData, width, height);

            var faces = _detector.DetectFaces(image);
            
            foreach (var face in faces)
            {
                float conf = face.Confidence ?? 0f;
                // Filter faces with low confidence to improve detection quality
                if (conf < 0.5f) continue;
                
                boxes.Add(new BoundingBox
                {
                    X = (int)Math.Max(0, face.Box.X),
                    Y = (int)Math.Max(0, face.Box.Y),
                    Width = (int)face.Box.Width,
                    Height = (int)face.Box.Height,
                    Confidence = conf,
                    Landmarks = face.Landmarks != null
                        ? face.Landmarks.Select(p => new System.Drawing.PointF(p.X, p.Y)).ToArray()
                        : null
                });
            }
            return boxes;
        }

        /// <summary>
        /// Releases the unmanaged resources used by the FaceDetector, 
        /// including the inference session and memory cache.
        /// </summary>
        public void Dispose()
        {
            (_detector as IDisposable)?.Dispose();
            _cache?.Dispose();
        }

    }
}
