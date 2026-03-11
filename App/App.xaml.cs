using System.Windows;
using Logging;

namespace FacialRecognitionApp;

/// <summary>
/// Global application class responsible for bootstrap, lifecycle management, and top-level exception handling.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Orchestrates application-wide events and state transitions.
///
/// Responsibilities:
/// - Initialize the shared logging infrastructure (AppLogger).
/// - Manage application-wide state transitions (Startup, Exit).
/// - Provide a robust safety net for unhandled exceptions.
///
/// Dependencies:
/// - Logging (AppLogger)
/// - Windows Presentation Foundation (WPF)
///
/// Architectural Role:
/// Application Entry Point / Bootstrapper.
/// </remarks>
public partial class App : Application
{
    /// <summary>
    /// Configures the application environment and subsystems during startup.
    /// </summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Initialize structured logger before anything else
        AppLogger.Initialize("logs/facialrecognition.log");
        AppLogger.Instance.Information("Application starting...");

        // Capture unhandled exceptions on UI and background threads
        DispatcherUnhandledException += (_, ex) =>
        {
            AppLogger.Instance.Fatal(ex.Exception, "Unhandled UI thread exception");
            ex.Handled = true;
        };

        AppDomain.CurrentDomain.UnhandledException += (_, ex) =>
        {
            AppLogger.Instance.Fatal(ex.ExceptionObject as Exception,
                "Unhandled background thread exception");
        };
    }

    /// <summary>
    /// Orchestrates a clean shutdown of the application.
    /// </summary>
    protected override void OnExit(ExitEventArgs e)
    {
        AppLogger.Instance.Information("Application shutting down.");
        AppLogger.CloseAndFlush();
        base.OnExit(e);
    }
}
