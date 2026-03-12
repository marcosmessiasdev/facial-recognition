using System.Diagnostics;
using System.Runtime.InteropServices;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.Core.Exceptions;
using FlaUI.UIA3;

namespace E2ETests;

/// <summary>
/// Common fixture: builds the app, launches it, exposes the FlaUI Application handle,
/// and tears down after the test.
/// </summary>
public abstract class AppTestBase
{
    private static int s_appBuilt;

#pragma warning disable NUnit1032 // Disposal is handled by [OneTimeTearDown] (CloseApp) and by explicit StopApp() in restart tests.
    private UIA3Automation? _automation;

    protected Application? App { get; private set; }
#pragma warning restore NUnit1032
    protected UIA3Automation Automation => _automation ?? throw new InvalidOperationException("Automation is not initialized yet.");
    protected Window? MainWindow { get; private set; }

    protected void RefreshMainWindow(int timeoutSeconds = 10)
    {
        if (App == null)
        {
            throw new InvalidOperationException("App is not initialized yet.");
        }

        MainWindow = App.GetMainWindow(Automation, TimeSpan.FromSeconds(Math.Clamp(timeoutSeconds, 1, 60)));
    }

    protected T UiRetry<T>(Func<Window, T> action, int attempts = 6, int delayMs = 250)
    {
        ArgumentNullException.ThrowIfNull(action);

        attempts = Math.Clamp(attempts, 1, 20);
        delayMs = Math.Clamp(delayMs, 0, 2000);

        Exception? last = null;
        for (int i = 0; i < attempts; i++)
        {
            try
            {
                RefreshMainWindow(timeoutSeconds: 10);
                if (MainWindow == null)
                {
                    throw new InvalidOperationException("Main window is not available.");
                }

                return action(MainWindow);
            }
            catch (COMException ex)
            {
                last = ex;
            }
            catch (ElementNotAvailableException ex)
            {
                last = ex;
            }

            if (i < attempts - 1 && delayMs > 0)
            {
                Thread.Sleep(delayMs);
            }
        }

        throw new InvalidOperationException("UIAutomation failed repeatedly while interacting with the app window.", last);
    }

    private static string FindAppExe()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("FACIAL_APP_EXE");
        if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
        {
            return fromEnv;
        }

        string repoRoot = FindRepoRoot(AppDomain.CurrentDomain.BaseDirectory);
        string exePath = Path.Combine(repoRoot, "src", "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "App.exe");
        return File.Exists(exePath) ? exePath : "NOT_FOUND";
    }

    protected static readonly string AppExe = FindAppExe();

    private static string FindRepoRoot(string startDir)
    {
        DirectoryInfo? cur = new(startDir);
        while (cur != null)
        {
            if (File.Exists(Path.Combine(cur.FullName, "src", "App", "App.csproj")))
            {
                return cur.FullName;
            }
            cur = cur.Parent;
        }

        throw new DirectoryNotFoundException("Repo root not found (expected 'src/App/App.csproj').");
    }

    private static void BuildApp(string repoRoot)
    {
        ProcessStartInfo psi = new("dotnet", "build src/App/App.csproj -c Debug")
        {
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        using Process p = Process.Start(psi) ?? throw new InvalidOperationException("Failed to start dotnet build.");
        string stdout = p.StandardOutput.ReadToEnd();
        string stderr = p.StandardError.ReadToEnd();
        _ = p.WaitForExit(10 * 60 * 1000);

        if (p.ExitCode != 0)
        {
            throw new InvalidOperationException("dotnet build src/App/App.csproj failed.\n" + stdout + "\n" + stderr);
        }
    }

    // Path to the log file written by Serilog (relative to current directory because Process.Start inherited it)
    protected static string LogFile
    {
        get
        {
            string repoRoot = FindRepoRoot(AppDomain.CurrentDomain.BaseDirectory);
            string logDir = Path.Combine(repoRoot, "src", "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "logs");
            return Path.Combine(logDir, $"facialrecognition{DateTime.Now:yyyyMMdd}.log");
        }
    }

    [OneTimeSetUp]
    public void LaunchApp()
    {
        StartApp(seedIdentity: true);
    }

    protected void StartApp(bool seedIdentity, string? audioTestFileOverride = null)
    {
        string repoRoot = FindRepoRoot(TestContext.CurrentContext.TestDirectory);
        if (Interlocked.CompareExchange(ref s_appBuilt, 1, 0) == 0)
        {
            BuildApp(repoRoot);
        }

        // Seed a deterministic identity DB in the app working directory so E2E can validate recognition.
        // Can be disabled by setting E2E_SEED_IDENTITY=false.
        string identityDb = Path.Combine(TestContext.CurrentContext.TestDirectory, "identity.db");
        Environment.SetEnvironmentVariable("identity_db", identityDb);

        if (seedIdentity)
        {
            string? seed = Environment.GetEnvironmentVariable("E2E_SEED_IDENTITY");
            bool doSeed = !string.Equals(seed, "false", StringComparison.OrdinalIgnoreCase) &&
                          !string.Equals(seed, "0", StringComparison.OrdinalIgnoreCase);
            if (doSeed)
            {
                string arcface = Path.Combine(repoRoot, "src", "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "onnx", "arcface.onnx");
                string scrfd = Path.Combine(repoRoot, "src", "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "onnx", "scrfd_2.5g_kps.onnx");
                if (File.Exists(arcface) && File.Exists(scrfd))
                {
                    IdentitySeed.SeedIdentityDb(TestContext.CurrentContext.TestDirectory, scrfd, arcface);
                }
            }
        }

        // Re-evaluate at runtime: build happens in OneTimeSetUp, so the exe might not exist during static init.
        string? fromEnv = Environment.GetEnvironmentVariable("FACIAL_APP_EXE");
        string exePath = !string.IsNullOrWhiteSpace(fromEnv) && File.Exists(fromEnv)
            ? fromEnv
            : FindAppExe();

        Assert.That(File.Exists(exePath),
            $"Application binary not found: {exePath}\n" +
            "Or set FACIAL_APP_EXE env var to the full path of App.exe");

        try
        {
            _ = Directory.CreateDirectory(Path.GetDirectoryName(LogFile)!);
            if (File.Exists(LogFile))
            {
                File.Delete(LogFile);
            }
        }
        catch
        {
            // Ignore: if we can't clean logs, tests still run; assertions should avoid relying on old content.
        }

        _automation = new UIA3Automation();

        // Default E2E configuration (offline + deterministic audio injection).
        // These can be overridden by environment variables already set on the machine.
        string testAudio = audioTestFileOverride ??
                           Path.Combine(TestContext.CurrentContext.TestDirectory, "audio", "e2e_fixture_10_words.wav");

        ProcessStartInfo psi = new(exePath)
        {
            WorkingDirectory = TestContext.CurrentContext.TestDirectory,
            UseShellExecute = false
        };

        if (Environment.GetEnvironmentVariable("audio_source") == null)
        {
            psi.EnvironmentVariables["audio_source"] = "test";
        }

        if (Environment.GetEnvironmentVariable("enable_audio_vad") == null)
        {
            psi.EnvironmentVariables["enable_audio_vad"] = "true";
        }

        if (Environment.GetEnvironmentVariable("audio_vad_speech_threshold") == null)
        {
            psi.EnvironmentVariables["audio_vad_speech_threshold"] = "0.005";
        }

        if (Environment.GetEnvironmentVariable("audio_vad_min_speech_ms") == null)
        {
            psi.EnvironmentVariables["audio_vad_min_speech_ms"] = "0";
        }

        if (Environment.GetEnvironmentVariable("audio_vad_min_silence_ms") == null)
        {
            psi.EnvironmentVariables["audio_vad_min_silence_ms"] = "120";
        }

        if (Environment.GetEnvironmentVariable("audio_vad_hangover_ms") == null)
        {
            psi.EnvironmentVariables["audio_vad_hangover_ms"] = "1500";
        }

        if (Environment.GetEnvironmentVariable("enable_transcription") == null)
        {
            psi.EnvironmentVariables["enable_transcription"] = "true";
        }

        if (Environment.GetEnvironmentVariable("transcription_min_segment_ms") == null)
        {
            psi.EnvironmentVariables["transcription_min_segment_ms"] = "400";
        }

        if (Environment.GetEnvironmentVariable("transcription_hangover_ms") == null)
        {
            // Longer hangover reduces micro-segments and makes STT/diarization more reliable in E2E.
            psi.EnvironmentVariables["transcription_hangover_ms"] = "1500";
        }

        if (Environment.GetEnvironmentVariable("enable_speaker_diarization") == null)
        {
            psi.EnvironmentVariables["enable_speaker_diarization"] = "true";
        }

        if (Environment.GetEnvironmentVariable("diarization_window_ms") == null)
        {
            // Keep windows short so even brief speech bursts produce embeddings deterministically.
            psi.EnvironmentVariables["diarization_window_ms"] = "500";
        }

        if (Environment.GetEnvironmentVariable("diarization_hop_ms") == null)
        {
            psi.EnvironmentVariables["diarization_hop_ms"] = "250";
        }

        if (Environment.GetEnvironmentVariable("audio_test_file") == null)
        {
            psi.EnvironmentVariables["audio_test_file"] = testAudio;
        }

        if (Environment.GetEnvironmentVariable("audio_test_loop") == null)
        {
            psi.EnvironmentVariables["audio_test_loop"] = "true";
        }

        if (Environment.GetEnvironmentVariable("audio_test_initial_silence_ms") == null)
        {
            psi.EnvironmentVariables["audio_test_initial_silence_ms"] = "0";
        }

        if (Environment.GetEnvironmentVariable("transcription_language") == null)
        {
            psi.EnvironmentVariables["transcription_language"] = "en";
        }

        // Make recognition less brittle across small crop/alignment differences in the offline fixture.
        if (Environment.GetEnvironmentVariable("recognition_threshold") == null)
        {
            psi.EnvironmentVariables["recognition_threshold"] = "0.30";
        }

        // Ensure the app reads/writes identity DB to the deterministic E2E location.
        if (Environment.GetEnvironmentVariable("identity_db") != null)
        {
            psi.EnvironmentVariables["identity_db"] = Environment.GetEnvironmentVariable("identity_db")!;
        }

        App = Application.Launch(psi);
        _ = App.WaitWhileMainHandleIsMissing(TimeSpan.FromSeconds(10));

        MainWindow = App.GetMainWindow(Automation, TimeSpan.FromSeconds(10));
        Assert.That(MainWindow, Is.Not.Null, "Main window did not appear within 10s");
    }

    [OneTimeTearDown]
    public void CloseApp()
    {
        StopApp();
    }

    protected void StopApp()
    {
        try { _ = (App?.Close()); } catch { /* ignore */ }
        try { App?.Dispose(); } catch { /* ignore */ }
        App = null;
        MainWindow = null;

        try { _automation?.Dispose(); } catch { /* ignore */ }
        _automation = null;
    }

    protected static string ReadLatestLog()
    {
        if (!File.Exists(LogFile))
        {
            return "";
        }
        // Copy to temp — file may be locked by Serilog sink
        string tmp = Path.GetTempFileName();
        File.Copy(LogFile, tmp, overwrite: true);
        return File.ReadAllText(tmp);
    }

    protected static void WaitForLogContains(string text, int timeoutMs = 8000, int pollMs = 200)
    {
        Stopwatch sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < timeoutMs)
        {
            string log = ReadLatestLog();
            if (log.Contains(text, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            Thread.Sleep(pollMs);
        }

        string last = ReadLatestLog();
        string tail = last.Length <= 4000 ? last : last[^4000..];
        Assert.Fail($"Timed out waiting for log to contain: '{text}'.\n--- log tail ---\n{tail}");
    }

    protected static void WaitForLogContainsAny(string[] texts, int timeoutMs = 8000, int pollMs = 200)
    {
        Stopwatch sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < timeoutMs)
        {
            string log = ReadLatestLog();
            if (texts.Any(t => log.Contains(t, StringComparison.OrdinalIgnoreCase)))
            {
                return;
            }

            Thread.Sleep(pollMs);
        }

        string last = ReadLatestLog();
        string tail = last.Length <= 4000 ? last : last[^4000..];
        Assert.Fail($"Timed out waiting for log to contain any of: '{string.Join("' | '", texts)}'.\n--- log tail ---\n{tail}");
    }
}
