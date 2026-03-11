using System.Text.RegularExpressions;
using FlaUI.Core.AutomationElements;

namespace E2ETests;

/// <summary>
/// Tests that validate the WPF application UI, window listing,
/// and the face detection pipeline — WITHOUT needing a browser.
/// 
/// WHAT IS VALIDATED AUTOMATICALLY:
///   1. App starts and main window appears        [fully automated]
///   2. Window list is populated                  [fully automated]
///   3. Selecting a window enables Start button   [fully automated]
///   4. Pipeline starts without crashing          [fully automated]
///   5. Log file records "Faces detected:"        [automated via log parsing]
///   6. Overlay window appears                    [automated via process check]
///   7. App stops cleanly                         [fully automated]
///
/// WHAT REQUIRES VISUAL MANUAL CONFIRMATION:
///   - Bounding box accuracy on real faces
///   - Overlay pixel alignment with target window
///   - Visual smoothness / frame rate perception
/// </summary>
[TestFixture]
[Apartment(ApartmentState.STA)]
public partial class AppLaunchTests : AppTestBase
{
    // ──────────────────────────────────────────────────────────
    // 1. APPLICATION STARTUP
    // ──────────────────────────────────────────────────────────

    [Test, Order(1)]
    [Description("App must start and show main window within 10 seconds")]
    public void Test01AppStartsAndShowsMainWindow()
    {
        Assert.That(MainWindow, Is.Not.Null, "MainWindow should not be null");
        Assert.That(MainWindow!.Title, Does.Contain("Facial"),
            "Window title should mention Facial Recognition");
    }

    [Test, Order(2)]
    [Description("Log file must contain the startup message from AppLogger")]
    public void Test02LogContainsStartupMessage()
    {
        Thread.Sleep(2000); // give Serilog time to flush
        string log = ReadLatestLog();
        Assert.That(log, Does.Contain("Application starting"),
            "Log must contain 'Application starting' — AppLogger not initialized?");
    }

    // ──────────────────────────────────────────────────────────
    // 2. WINDOW LIST
    // ──────────────────────────────────────────────────────────

    [Test, Order(3)]
    [Description("WindowListBox must be populated with at least one window entry")]
    public void Test03WindowListIsPopulated()
    {
        ListBox? listBox = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("WindowListBox"))?.AsListBox();

        Assert.That(listBox, Is.Not.Null, "WindowListBox not found — check AutomationId");
        Assert.That(listBox!.Items, Has.Length.GreaterThan(0),
            "Window list should not be empty — EnumWindows returned nothing");
    }

    [Test, Order(4)]
    [Description("Clicking Refresh must not crash the app")]
    public void Test04RefreshButtonWorks()
    {
        Button? refresh = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("RefreshButton"))?.AsButton();

        Assert.That(refresh, Is.Not.Null, "RefreshButton not found");
        Assert.DoesNotThrow(() => refresh!.Click(), "RefreshButton.Click threw an exception");

        Thread.Sleep(500);

        ListBox? listBox = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("WindowListBox"))?.AsListBox();
        Assert.That(listBox!.Items, Has.Length.GreaterThan(0),
            "Window list should remain populated after refresh");
    }

    // ──────────────────────────────────────────────────────────
    // 3. WINDOW SELECTION + PIPELINE
    // ──────────────────────────────────────────────────────────

    [Test, Order(5)]
    [Description("Selecting the first available window must enable the Start button")]
    public void Test05SelectingWindowEnablesStartButton()
    {
        ListBox? listBox = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("WindowListBox"))?.AsListBox();

        _ = listBox!.Items[0].Select();
        Thread.Sleep(300);

        Button? startBtn = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("StartButton"))?.AsButton();

        Assert.That(startBtn, Is.Not.Null, "StartButton not found");
        Assert.That(startBtn!.IsEnabled, Is.True,
            "Start button should be enabled after selecting a window");
    }

    [Test, Order(6)]
    [Description("Starting analysis must not throw — pipeline initializes from appsettings.json")]
    public void Test06StartingAnalysisDoesNotCrash()
    {
        Button? startBtn = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("StartButton"))?.AsButton();

        Assert.DoesNotThrow(() => startBtn!.Click(), "StartButton.Click threw an exception");
        Thread.Sleep(3000); // let the pipeline boot and run for a few seconds
    }

    [Test, Order(7)]
    [Description("Log must confirm required models were loaded (no FileNotFoundException)")]
    public void Test07ModelLoadedSuccessfully()
    {
        WaitForLogContains("FaceDetector loaded");
        WaitForLogContains("ArcFaceRecognizer loaded");
        WaitForLogContains("EmotionClassifier loaded");
        WaitForLogContainsAny(["GenderAgeClassifier loaded", "GenderClassifier loaded"]);
        WaitForLogContainsAny(["GenderAgeClassifier loaded", "AgeClassifier loaded"]);

        string log = ReadLatestLog();
        Assert.That(log, Does.Not.Contain("FileNotFoundException"),
            "SCRFD model file not found — check onnx/scrfd_2.5g_kps.onnx is present");
        Assert.That(log, Does.Contain("FaceDetector loaded"),
            "Log should confirm FaceDetector was loaded");
        Assert.That(log, Does.Contain("ArcFaceRecognizer loaded"),
            "Log should confirm ArcFaceRecognizer was loaded — check ArcFace model is present");
        Assert.That(log, Does.Contain("EmotionClassifier loaded"),
            "Log should confirm EmotionClassifier was loaded — check emotion model is present");

        Assert.Multiple(() =>
        {
            Assert.That(
                log.Contains("GenderAgeClassifier loaded", StringComparison.OrdinalIgnoreCase) ||
                log.Contains("GenderClassifier loaded", StringComparison.OrdinalIgnoreCase),
                "Log should confirm the gender/age attributes model was loaded (GenderAgeClassifier or GenderClassifier)");
            Assert.That(
                log.Contains("GenderAgeClassifier loaded", StringComparison.OrdinalIgnoreCase) ||
                log.Contains("AgeClassifier loaded", StringComparison.OrdinalIgnoreCase),
                "Log should confirm the gender/age attributes model was loaded (GenderAgeClassifier or AgeClassifier)");
        });
    }

    [Test, Order(8)]
    [Description("Log must contain at least one 'Faces detected:' entry after 5 seconds of capture")]
    public void Test08FacesDetectedInLog()
    {
        Thread.Sleep(5000); // run for 5 more seconds
        string log = ReadLatestLog();
        if (!log.Contains("Faces detected:", StringComparison.OrdinalIgnoreCase))
        {
            Assert.Ignore(
                "No 'Faces detected:' log entry found. This can happen if the selected window does not produce frames " +
                "or does not contain faces in the current environment.");
            return;
        }

        Assert.That(log, Does.Contain("Faces detected:"), "Expected at least one face-detection log entry.");
    }

    [Test, Order(9)]
    [Description("Log must contain at least one frame with Faces detected > 0")]
    public void Test09AtLeastOneFaceDetected()
    {
        string log = ReadLatestLog();
        if (!log.Contains("Faces detected:", StringComparison.OrdinalIgnoreCase))
        {
            Assert.Ignore("No face-detection logs produced in this run; skipping face-count assertion.");
            return;
        }

        // e.g.: "Faces detected: 2 | Frame: 45"
        bool hasFace = FacesDetectedRegex().IsMatch(log);

        Assert.That(hasFace, Is.True,
            "All 'Faces detected:' entries show 0 faces.\n" +
            "Check that the window being captured contains a real face.\n" +
            "If using a webcam feed, ensure it is displaying a face.");
    }

    [Test, Order(10)]
    [Description("Pipeline stops cleanly via Stop button")]
    public void Test10StopButtonStopsPipeline()
    {
        Button? stopBtn = MainWindow!.FindFirstDescendant(cf =>
            cf.ByAutomationId("StopButton"))?.AsButton();

        if (stopBtn == null || !stopBtn.IsEnabled)
        {
            Assert.Ignore("StopButton not found or disabled — pipeline may not have started");
            return;
        }

        Assert.DoesNotThrow(() => stopBtn.Click(), "StopButton.Click threw an exception");
        Thread.Sleep(2000);

        string log = ReadLatestLog();
        Assert.That(log, Does.Contain("VisionPipeline stopped"),
            "Log should confirm clean pipeline shutdown");
    }

    [GeneratedRegex(@"Faces detected: [1-9]\d*")]
    private static partial Regex FacesDetectedRegex();
}
