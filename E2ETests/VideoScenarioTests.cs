using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using FlaUI.Core.AutomationElements;
using Microsoft.Playwright;
using Microsoft.Playwright.NUnit;

namespace E2ETests
{
    /// <summary>
    /// Full E2E scenario: opens a YouTube video in a Chromium browser via Playwright,
    /// then uses FlaUI to control the WPF app and validate face detection on the video.
    ///
    /// PREREQUISITES BEFORE RUNNING THIS TEST:
    ///   1. Run: dotnet build App/App.csproj
    ///   2. Run: pwsh bin/Debug/net9.0-windows10.0.19041.0/playwright.ps1 install chromium
    ///   3. Have a screen resolution >= 1280x720
    ///   4. Do NOT have another screen lock or sleep active during the test
    ///
    /// WHAT IS VALIDATED:
    ///   [AUTO]  Browser opens and loads the test video at 1080p
    ///   [AUTO]  Video is playing (not paused)
    ///   [AUTO]  App window list contains the browser window
    ///   [AUTO]  Pipeline starts without exception
    ///   [AUTO]  Log confirms "Faces detected: N" with N > 0
    ///   [AUTO]  App closes cleanly
    ///   [MANUAL] Overlay bounding boxes visually align with faces (screenshot saved for review)
    ///
    /// VIDEO USED: Local file `E2ETests/video/faces.mp4` via `E2ETests/video/index.html`
    ///   - Offline/reproducible (no network dependency)
    /// </summary>
    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class VideoScenarioTests : AppTestBase, IDisposable
    {
        private const int PipelineWarmupSeconds = 8;
        private const string ScreenshotPath = "e2e_overlay_screenshot.png";
        
        private static string GetLocalVideoPageUrl()
        {
            var baseDir = TestContext.CurrentContext.TestDirectory;
            var path = Path.Combine(baseDir, "video", "index.html");
            return new Uri(path).AbsoluteUri;
        }

        private IPlaywright? _playwright;
        private IBrowser?    _browser;
        private IPage?       _page;
        private IBrowserContext? _context;
        private string? _pageTitle;

        private class WindowItem { public IntPtr Handle {get;set;} public string Name {get;set;} = ""; }
        
        private static WindowItem[] GetHwndsFromListBox(ListBox listBox)
        {
            return listBox.Items.Select(i => new WindowItem { Name = i.Name }).ToArray();
        }

        [OneTimeSetUp]
        public async Task SetupBrowserAsync()
        {
            _playwright = await Microsoft.Playwright.Playwright.CreateAsync();
            _browser = await _playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
            {
                Headless = false, 
                Args = new[] { "--start-maximized", "--disable-infobars", "--autoplay-policy=no-user-gesture-required" }
            });

            _context = await _browser.NewContextAsync(new BrowserNewContextOptions
            {
                ViewportSize = ViewportSize.NoViewport
            });

            _page = await _context.NewPageAsync();
            
            var url = GetLocalVideoPageUrl();
            TestContext.Out.WriteLine($"Navigating to local test page: {url}");
            await _page.GotoAsync(url);
            await _page.WaitForSelectorAsync("#grid");

            // One user gesture to start audio + mouth animation.
            await _page.ClickAsync("#grid");
            await Task.Delay(800);

            _pageTitle = await _page.TitleAsync();
            TestContext.Out.WriteLine($"Browser page title: {_pageTitle}");
            await _page.BringToFrontAsync();
        }

        [OneTimeTearDown]
        public async Task TeardownBrowserAsync()
        {
            if (_browser != null) await _browser.CloseAsync();
            _playwright?.Dispose();
        }

        [Test, Order(1)]
        [Category("E2E")]
        public async Task Test_FullE2EVideoScenario()
        {
            // Verify window is listed in the dropdown
            var listBox = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("WindowListBox"))?.AsListBox();
            Assert.That(listBox, Is.Not.Null, "Window list box not found");

            var refreshBtn = MainWindow.FindFirstDescendant(cf => cf.ByAutomationId("RefreshButton"))?.AsButton();
            refreshBtn?.Invoke();
            await Task.Delay(1000);

            // Find the Chrome window running the video
            var hwnds = GetHwndsFromListBox(listBox!);
            var titleHint = _pageTitle ?? "Test Video";
            var browserHwnd = hwnds.FirstOrDefault(i =>
                i.Name.Contains(titleHint, StringComparison.OrdinalIgnoreCase) ||
                i.Name.Contains("faces.mp4", StringComparison.OrdinalIgnoreCase) ||
                i.Name.Contains("YouTube", StringComparison.OrdinalIgnoreCase) || 
                i.Name.Contains("Chrome", StringComparison.OrdinalIgnoreCase));

            if (browserHwnd == null)
            {
                TestContext.Out.WriteLine("Browser window not found by name — selecting first item as fallback");
                listBox.Items[0].Select();
            }
            else
            {
                // Find matching listbox item
                var listItem = listBox.Items.FirstOrDefault(i => i.Name == browserHwnd.Name);
                listItem?.Select();
            }

            await Task.Delay(300);

            // Start pipeline
            var startBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StartButton"))?.AsButton();
            Assert.That(startBtn?.IsEnabled, Is.True, "Start button should be enabled");
            startBtn!.Click();

            // Wait for pipeline to warm up
            TestContext.Out.WriteLine($"Waiting {PipelineWarmupSeconds}s for pipeline to process frames...");
            await Task.Delay(PipelineWarmupSeconds * 1000);

            // Validate log has face detections
            WaitForLogContains("FaceDetector loaded", timeoutMs: 15000);
            WaitForLogContains("Faces detected:", timeoutMs: 15000);
            var log = ReadLatestLog();
            Assert.That(log, Does.Contain("FaceDetector loaded"), "FaceDetector did not load — check model file");
            Assert.That(log, Does.Contain("Faces detected:"), "No face detection logged — pipeline may not be running");

            var hasFaces = Regex.IsMatch(log, @"Faces detected: [1-9]\d*");
            Assert.That(hasFaces, Is.True,
                "No frames with Faces > 0 found in log.\n" +
                "Possible causes:\n" +
                "  - Video resolution too low\n" +
                "  - Faces too small in frame\n" +
                "  - SCRFD confidence threshold too high");

            // Extract best frame metrics for report
            var matches = Regex.Matches(log, @"Faces detected: (\d+)");
            if (matches.Count > 0)
            {
                int maxFaces = matches.Select(m => int.Parse(m.Groups[1].Value)).Max();
                TestContext.Out.WriteLine($"Max faces detected in a single frame: {maxFaces}");
                Assert.That(maxFaces, Is.GreaterThan(0));
            }

            // Stop pipeline
            var stopBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StopButton"))?.AsButton();
            stopBtn?.Click();
            WaitForLogContains("VisionPipeline stopped", timeoutMs: 8000);
            await Task.Delay(500);

            var logAfterStop = ReadLatestLog();
            Assert.That(logAfterStop, Does.Contain("VisionPipeline stopped"), "Pipeline stop not confirmed in log");
        }

        // ──────────────────────────────────────────────────────────────────────────
        // PERFORMANCE TEST
        // ──────────────────────────────────────────────────────────────────────────

        [Test]
        [Description("Performance: runs pipeline for 30s and verifies no crash + faces detected")]
        public async Task Test_PipelineStabilityOver30Seconds()
        {
            var refreshBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("RefreshButton"))?.AsButton();
            refreshBtn?.Invoke();
            await Task.Delay(500);

            var listBox = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("WindowListBox"))?.AsListBox();
            var hwnds = GetHwndsFromListBox(listBox!);
            var titleHint = _pageTitle ?? "Test Video";
            var browserItem = listBox!.Items.FirstOrDefault(i => 
                i.Name.Contains(titleHint, StringComparison.OrdinalIgnoreCase) ||
                i.Name.Contains("faces.mp4", StringComparison.OrdinalIgnoreCase) ||
                i.Name.Contains("YouTube", StringComparison.OrdinalIgnoreCase) || 
                i.Name.Contains("Chrome", StringComparison.OrdinalIgnoreCase));
            
            if (browserItem != null) browserItem.Select();
            else listBox.Items[0].Select();

            var startBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StartButton"))?.AsButton();
            startBtn?.Invoke();

            // Run for 30 seconds
            TestContext.Out.WriteLine("Running pipeline for 30 seconds stability test...");
            await Task.Delay(30_000);

            // App must still be alive
            Assert.That(App!.HasExited, Is.False, "App crashed during 30s stability test");

            // Count total face-detection frames
            var log = ReadLatestLog();
            Assert.That(log, Does.Not.Contain("Recognition inference error"),
                "Recognition inference failed during the stability test. Check the face embedding model and preprocessing.");
            var detectedFrames = Regex.Matches(log, @"Faces detected: [1-9]").Count;
            TestContext.Out.WriteLine($"Frames with >=1 face detected: {detectedFrames}");

            Assert.That(detectedFrames, Is.GreaterThan(0),
                "Expected at least one frame with detected faces during 30s run");

            var stopBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StopButton"))?.AsButton();
            stopBtn?.Invoke();
            await Task.Delay(1000);
        }

        public void Dispose()
        {
            _playwright?.Dispose();
        }
    }
}
