using System.Windows;
using System.IO;
using Logging;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Config;
using VisionEngine;

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
    private IHost? _host;

    public static IServiceProvider Services =>
        (Current as App)?._host?.Services ?? throw new InvalidOperationException("Host is not initialized yet.");

    /// <summary>
    /// Configures the application environment and subsystems during startup.
    /// </summary>
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        // Initialize structured logger before anything else
        string basedir = AppDomain.CurrentDomain.BaseDirectory;
        string logPath = Path.Combine(basedir, "logs", "facialrecognition.log");
        AppLogger.Initialize(logPath);
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

        _host = Host.CreateDefaultBuilder()
            .ConfigureAppConfiguration((_, cfg) =>
            {
                // Ensure config is loaded from the app folder (portable, no machine-specific paths).
                cfg.Sources.Clear();
                cfg.SetBasePath(basedir);
                cfg.AddJsonFile("appsettings.json", optional: true, reloadOnChange: false);
                cfg.AddEnvironmentVariables();
            })
            .ConfigureServices((ctx, services) =>
            {
                // Configuration
                AppConfig appConfig = AppConfig.Load();
                services.AddSingleton(appConfig);
                services.AddSingleton<IOptions<AppConfig>>(_ => Options.Create(appConfig));

                services.AddVisionEngineServices();

                // UI
                services.AddSingleton<MainWindow>();
            })
            .Build();

        _host.Start();

        MainWindow main = _host.Services.GetRequiredService<MainWindow>();
        MainWindow = main;
        main.Show();
    }

    /// <summary>
    /// Orchestrates a clean shutdown of the application.
    /// </summary>
    protected override void OnExit(ExitEventArgs e)
    {
        AppLogger.Instance.Information("Application shutting down.");

        try
        {
            _host?.StopAsync(TimeSpan.FromSeconds(3)).GetAwaiter().GetResult();
            _host?.Dispose();
            _host = null;
        }
        catch
        {
            // ignore best-effort shutdown
        }

        AppLogger.CloseAndFlush();
        base.OnExit(e);
    }
}
