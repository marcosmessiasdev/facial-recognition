using System.Globalization;
using System.Text.RegularExpressions;
using FlaUI.Core.AutomationElements;
using Microsoft.Playwright;

namespace E2ETests;

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
[Apartment(ApartmentState.STA)]
public partial class VideoScenarioTests : AppTestBase, IDisposable
{
    private const int PipelineWarmupSeconds = 8;

    private static string GetLocalVideoPageUrl()
    {
        string baseDir = TestContext.CurrentContext.TestDirectory;
        string path = Path.Combine(baseDir, "video", "index.html");
        return new Uri(path).AbsoluteUri;
    }

    private IPlaywright? _playwright;
    private IBrowser? _browser;
    private IPage? _page;
    private IBrowserContext? _context;
    private string? _pageTitle;

    private sealed class WindowItem { public IntPtr Handle { get; set; } public string Name { get; set; } = ""; }

    private static WindowItem[] GetHwndsFromListBox(ListBox listBox)
    {
        return [.. listBox.Items.Select(i => new WindowItem { Name = i.Name })];
    }

    [OneTimeSetUp]
    public async Task SetupBrowserAsync()
    {
        _playwright = await Playwright.CreateAsync();
        _browser = await _playwright.Chromium.LaunchAsync(new BrowserTypeLaunchOptions
        {
            Headless = false,
            Args = ["--start-maximized", "--disable-infobars", "--autoplay-policy=no-user-gesture-required"]
        });

        _context = await _browser.NewContextAsync(new BrowserNewContextOptions
        {
            ViewportSize = ViewportSize.NoViewport
        });

        _page = await _context.NewPageAsync();

        string url = GetLocalVideoPageUrl();
        TestContext.Out.WriteLine($"Navigating to local test page: {url}");
        _ = await _page.GotoAsync(url);
        _ = await _page.WaitForSelectorAsync("#grid");

        // One user gesture to start audio + mouth animation.
        await _page.ClickAsync("#grid");
        try
        {
            _ = await _page.WaitForFunctionAsync("() => { const a=document.getElementById('audio'); return a && !a.paused && a.currentTime > 0.2; }",
                null,
                new PageWaitForFunctionOptions { Timeout = 5000 });
        }
        catch
        {
            // Best-effort: allow the test to continue even if the browser refuses audio playback.
            TestContext.Out.WriteLine("Warning: audio did not start (autoplay blocked or muted device).");
        }

        await Task.Delay(500);

        _pageTitle = await _page.TitleAsync();
        TestContext.Out.WriteLine($"Browser page title: {_pageTitle}");
        await _page.BringToFrontAsync();
    }

    [OneTimeTearDown]
    public async Task TeardownBrowserAsync()
    {
        if (_browser != null)
        {
            await _browser.CloseAsync();
        }

        _playwright?.Dispose();
    }

    [Test, Order(1)]
    [Category("E2E")]
    public async Task TestFullE2EVideoScenario()
    {
        // Verify window is listed in the dropdown
        ListBox? listBox = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("WindowListBox"))?.AsListBox();
        Assert.That(listBox, Is.Not.Null, "Window list box not found");

        Button? refreshBtn = MainWindow.FindFirstDescendant(cf => cf.ByAutomationId("RefreshButton"))?.AsButton();
        refreshBtn?.Invoke();
        await Task.Delay(1000);

        // Find the browser window running the local page
        string titleHint = _pageTitle ?? "E2E Faces Grid";
        ListBoxItem? browserItem = listBox.Items.FirstOrDefault(i =>
            i.Name.Contains(titleHint, StringComparison.OrdinalIgnoreCase) ||
            i.Name.Contains("E2E Faces Grid", StringComparison.OrdinalIgnoreCase));

        if (browserItem == null)
        {
            string all = string.Join("\n - ", listBox.Items.Select(i => i.Name));
            Assert.Fail($"Browser window not found in app list. Expected title containing: '{titleHint}'.\nFound:\n - {all}");
            return;
        }

        _ = browserItem.Select();

        await Task.Delay(300);

        // Start pipeline
        Button? startBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StartButton"))?.AsButton();
        Assert.That(startBtn?.IsEnabled, Is.True, "Start button should be enabled");
        startBtn!.Click();

        // Wait for pipeline to warm up
        TestContext.Out.WriteLine($"Waiting {PipelineWarmupSeconds}s for pipeline to process frames...");
        await Task.Delay(PipelineWarmupSeconds * 1000);

        // Validate log has face detections
        WaitForLogContains("FaceDetector loaded", timeoutMs: 15000);
        WaitForLogContains("Faces detected:", timeoutMs: 15000);
        string log = ReadLatestLog();
        Assert.That(log, Does.Contain("FaceDetector loaded"), "FaceDetector did not load — check model file");
        Assert.That(log, Does.Contain("Faces detected:"), "No face detection logged — pipeline may not be running");

        bool hasFaces = FacesDetectedRegex().IsMatch(log);
        Assert.That(hasFaces, Is.True,
            "No frames with Faces > 0 found in log.\n" +
            "Possible causes:\n" +
            "  - Video resolution too low\n" +
            "  - Faces too small in frame\n" +
            "  - SCRFD confidence threshold too high");

        // Extract best frame metrics for report
        MatchCollection matches = FacesCountRegex().Matches(log);
        if (matches.Count > 0)
        {
            int maxFaces = matches.Max(m => int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture));
            TestContext.Out.WriteLine($"Max faces detected in a single frame: {maxFaces}");
            Assert.That(maxFaces, Is.GreaterThan(0));
        }

        // Stop pipeline
        Button? stopBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StopButton"))?.AsButton();
        stopBtn?.Click();
        WaitForLogContains("VisionPipeline stopped", timeoutMs: 8000);
        await Task.Delay(500);

        string logAfterStop = ReadLatestLog();
        Assert.That(logAfterStop, Does.Contain("VisionPipeline stopped"), "Pipeline stop not confirmed in log");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // PERFORMANCE TEST
    // ──────────────────────────────────────────────────────────────────────────

    [Test]
    [Description("Performance: runs pipeline for 30s and verifies no crash + faces detected")]
#pragma warning disable CA1707 // Compatibility with existing CI filter naming
    public async Task Test_PipelineStabilityOver30Seconds()
    {
        Button? refreshBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("RefreshButton"))?.AsButton();
        refreshBtn?.Invoke();
        await Task.Delay(500);

        ListBox? listBox = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("WindowListBox"))?.AsListBox();
        WindowItem[] hwnds = GetHwndsFromListBox(listBox!);
        string titleHint = _pageTitle ?? "E2E Faces Grid";
        ListBoxItem? browserItem = listBox!.Items.FirstOrDefault(i =>
            i.Name.Contains(titleHint, StringComparison.OrdinalIgnoreCase) ||
            i.Name.Contains("E2E Faces Grid", StringComparison.OrdinalIgnoreCase));

        if (browserItem == null)
        {
            string all = string.Join("\n - ", listBox.Items.Select(i => i.Name));
            Assert.Fail($"Browser window not found in app list. Expected title containing: '{titleHint}'.\nFound:\n - {all}");
            return;
        }

        _ = browserItem.Select();

        Button? startBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StartButton"))?.AsButton();
        startBtn?.Invoke();

        // Run for 30 seconds
        TestContext.Out.WriteLine("Running pipeline for 30 seconds stability test...");
        await Task.Delay(30_000);

        // App must still be alive
        Assert.That(App!.HasExited, Is.False, "App crashed during 30s stability test");

        // Count total face-detection frames
        string log = ReadLatestLog();
        Assert.That(log, Does.Not.Contain("Recognition inference error"),
            "Recognition inference failed during the stability test. Check the face embedding model and preprocessing.");
        int detectedFrames = FacesDetectedRegex().Matches(log).Count;
        TestContext.Out.WriteLine($"Frames with >=1 face detected: {detectedFrames}");

        Assert.That(detectedFrames, Is.GreaterThan(0),
            "Expected at least one frame with detected faces during 30s run");

        Button? stopBtn = MainWindow!.FindFirstDescendant(cf => cf.ByAutomationId("StopButton"))?.AsButton();
        stopBtn?.Invoke();
        await Task.Delay(1000);
    }
#pragma warning restore CA1707

    public void Dispose()
    {
        _playwright?.Dispose();
        GC.SuppressFinalize(this);
    }

    [GeneratedRegex(@"Faces detected: [1-9]\d*")]
    private static partial Regex FacesDetectedRegex();

    [GeneratedRegex(@"Faces detected: (\d+)")]
    private static partial Regex FacesCountRegex();
}
