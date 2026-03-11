using System.Diagnostics;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;

namespace E2ETests;

/// <summary>
/// Common fixture: builds the app, launches it, exposes the FlaUI Application handle,
/// and tears down after the test.
/// </summary>
public abstract class AppTestBase
{
    private Application? _app;
    private UIA3Automation? _automation;
    private Window? _mainWindow;

    protected Application? App => _app;
    protected UIA3Automation Automation => _automation ?? throw new InvalidOperationException("Automation is not initialized yet.");
    protected Window? MainWindow => _mainWindow;

    private static string FindAppExe()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("FACIAL_APP_EXE");
        if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
        {
            return fromEnv;
        }

        string baseDir = AppDomain.CurrentDomain.BaseDirectory;
        string pathRelative = Path.GetFullPath(Path.Combine(baseDir,
            "..", "..", "..", "..",
            "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "App.exe"));

        return File.Exists(pathRelative) ? pathRelative : "NOT_FOUND";
    }

    protected static readonly string AppExe = FindAppExe();

    // Path to the log file written by Serilog (relative to current directory because Process.Start inherited it)
    protected static string LogFile =>
        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "logs",
            $"facialrecognition{DateTime.Now:yyyyMMdd}.log");

    [OneTimeSetUp]
    public void LaunchApp()
    {
        // Re-evaluate at runtime in case env var was set after static init
        string exePath = Environment.GetEnvironmentVariable("FACIAL_APP_EXE") ?? AppExe;

        Assert.That(File.Exists(exePath),
            $"Application binary not found: {exePath}\n" +
            "Run 'dotnet build App/App.csproj' first.\n" +
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
        _app = Application.Launch(exePath);
        _ = _app.WaitWhileMainHandleIsMissing(TimeSpan.FromSeconds(10));

        _mainWindow = _app.GetMainWindow(Automation, TimeSpan.FromSeconds(10));
        Assert.That(_mainWindow, Is.Not.Null, "Main window did not appear within 10s");
    }

    [OneTimeTearDown]
    public void CloseApp()
    {
        try { _ = (_app?.Close()); } catch { /* ignore */ }
        _app?.Dispose();
        _automation?.Dispose();
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
    }
}
