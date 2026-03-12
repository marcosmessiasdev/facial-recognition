using FaceDetection;
using FaceRecognition;
using FramePipeline;
using IdentityStore;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace E2ETests;

internal static class IdentitySeed
{
    public static void SeedIdentityDb(string workDir, string scrfdModelPath, string arcFaceModelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workDir);
        ArgumentException.ThrowIfNullOrWhiteSpace(scrfdModelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(arcFaceModelPath);

        string identityDb = Path.Combine(workDir, "identity.db");
        string? prevIdentityDb = Environment.GetEnvironmentVariable("identity_db");
        Environment.SetEnvironmentVariable("identity_db", identityDb);
        try
        {
            if (File.Exists(identityDb))
            {
                File.Delete(identityDb);
            }
        }
        catch
        {
            // Best-effort: if we can't clean, seeding might still work (will append).
        }

        string prevCwd = Directory.GetCurrentDirectory();
        Directory.SetCurrentDirectory(workDir);
        try
        {
            using FaceDetector detector = new(scrfdModelPath);
            using ArcFaceRecognizer recognizer = new(arcFaceModelPath);
            using PersonRepository repo = new();

            // Seed from the static face tiles used by the offline E2E HTML fixture.
            // This avoids codec brittleness (mp4 decoding) and makes the test fully deterministic.
            (string File, string Name)[] seeds =
            {
                ("face1.jpg", "Neil"),
                ("face2.jpg", "Buzz"),
                ("face3.jpg", "Sally"),
                ("face4.jpg", "Mae")
            };

            string assetsDir = Path.Combine(workDir, "video", "assets");

            foreach ((string file, string name) in seeds)
            {
                string path = Path.Combine(assetsDir, file);
                if (!File.Exists(path))
                {
                    continue;
                }

                using Mat bgr = Cv2.ImRead(path, ImreadModes.Color);
                if (bgr.Empty())
                {
                    continue;
                }

                using Mat bgra = new();
                Cv2.CvtColor(bgr, bgra, ColorConversionCodes.BGR2BGRA);

                int stride = (int)bgra.Step();
                byte[] data = new byte[stride * bgra.Rows];
                Marshal.Copy(bgra.Data, data, 0, data.Length);
                using VisionFrame vf = new(data, bgra.Cols, bgra.Rows, stride);

                List<FacialRecognition.Domain.BoundingBox> boxes = detector.Detect(vf);
                if (boxes.Count == 0)
                {
                    continue;
                }

                // Pick the largest face in the image.
                FacialRecognition.Domain.BoundingBox best = boxes
                    .OrderByDescending(bx => bx.Width * bx.Height)
                    .First();

                Rect rect = new(
                    Math.Max(0, best.X),
                    Math.Max(0, best.Y),
                    Math.Max(1, Math.Min(best.Width, vf.Mat.Width - best.X)),
                    Math.Max(1, Math.Min(best.Height, vf.Mat.Height - best.Y)));

                if (rect.Width < 16 || rect.Height < 16)
                {
                    continue;
                }

                using Mat crop = new(vf.Mat, rect);
                float[] emb = recognizer.GetEmbedding(crop);
                repo.RegisterPerson(name, emb);
            }

            // Hard fail if the DB ended up empty: recognition E2E would be meaningless.
            int count = repo.GetAll().Count;
            if (count <= 0)
            {
                throw new InvalidOperationException("Identity seeding produced 0 persons. Check E2E fixture assets and SCRFD/ArcFace models.");
            }
        }
        finally
        {
            Directory.SetCurrentDirectory(prevCwd);
            Environment.SetEnvironmentVariable("identity_db", prevIdentityDb);
        }
    }
}
