using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using FlaUI.Core;
using FlaUI.Core.AutomationElements;
using FlaUI.UIA3;

namespace E2ETests
{
    /// <summary>
    /// Common fixture: builds the app, launches it, exposes the FlaUI Application handle,
    /// and tears down after the test.
    /// </summary>
    public abstract class AppTestBase
    {
        protected Application? App;
        protected UIA3Automation Automation = null!;
        protected Window? MainWindow;

        private static string FindAppExe()
        {
            var fromEnv = Environment.GetEnvironmentVariable("FACIAL_APP_EXE");
            if (!string.IsNullOrEmpty(fromEnv) && File.Exists(fromEnv))
                return fromEnv;

            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var pathRelative = Path.GetFullPath(Path.Combine(baseDir,
                "..", "..", "..", "..", 
                "App", "bin", "Debug", "net8.0-windows10.0.19041.0", "App.exe"));

            if (File.Exists(pathRelative)) return pathRelative;

            return "NOT_FOUND";
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
            var exePath = Environment.GetEnvironmentVariable("FACIAL_APP_EXE") ?? AppExe;

            Assert.That(File.Exists(exePath),
                $"Application binary not found: {exePath}\n" +
                "Run 'dotnet build App/App.csproj' first.\n" +
                "Or set FACIAL_APP_EXE env var to the full path of App.exe");

            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(LogFile)!);
                if (File.Exists(LogFile)) File.Delete(LogFile);
            }
            catch
            {
                // Ignore: if we can't clean logs, tests still run; assertions should avoid relying on old content.
            }

            Automation = new UIA3Automation();
            App = Application.Launch(exePath);
            App.WaitWhileMainHandleIsMissing(TimeSpan.FromSeconds(10));

            MainWindow = App.GetMainWindow(Automation, TimeSpan.FromSeconds(10));
            Assert.That(MainWindow, Is.Not.Null, "Main window did not appear within 10s");
        }

        [OneTimeTearDown]
        public void CloseApp()
        {
            try { App?.Close(); } catch { /* ignore */ }
            Automation?.Dispose();
        }

        protected string ReadLatestLog()
        {
            if (!File.Exists(LogFile)) return "";
            // Copy to temp — file may be locked by Serilog sink
            var tmp = Path.GetTempFileName();
            File.Copy(LogFile, tmp, overwrite: true);
            return File.ReadAllText(tmp);
        }

        protected void WaitForLogContains(string text, int timeoutMs = 8000, int pollMs = 200)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                var log = ReadLatestLog();
                if (log.Contains(text, StringComparison.OrdinalIgnoreCase))
                    return;
                Thread.Sleep(pollMs);
            }
        }

        protected void WaitForLogContainsAny(string[] texts, int timeoutMs = 8000, int pollMs = 200)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                var log = ReadLatestLog();
                if (texts.Any(t => log.Contains(t, StringComparison.OrdinalIgnoreCase)))
                    return;
                Thread.Sleep(pollMs);
            }
        }
    }
}
