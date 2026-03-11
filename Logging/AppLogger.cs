using Serilog;
using System.Globalization;

namespace Logging;

/// <summary>
/// Provides central configuration and access for the application's logging infrastructure.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Standardizes how logs are captured, formatted, and persisted throughout the system using Serilog.
///
/// Responsibilities:
/// - Configure logging sinks (Console and rolling File).
/// - Set minimum log levels and output templates.
/// - Expose a global ILogger instance for the entire application.
///
/// Dependencies:
/// - Serilog (Core logging library)
///
/// Architectural Role:
/// Infrastructure Component / Cross-cutting Concern.
///
/// Constraints:
/// - Should be initialized once at application startup.
/// </remarks>
public static class AppLogger
{
    /// <summary>
    /// Gets the singleton logger instance used throughout the application.
    /// </summary>
    /// <value>
    /// An <see cref="ILogger"/> implementing Serilog's interface. 
    /// defaults to the global <see cref="Log.Logger"/>.
    /// </value>
    public static ILogger Instance { get; private set; } = Log.Logger;

    /// <summary>
    /// Configures the logging pipeline with sinks, formatters, and rotation policies.
    /// </summary>
    /// <param name="logFilePath">
    /// The filesystem path where log files will be saved. 
    /// Supports daily rotation automatically.
    /// </param>
    public static void Initialize(string logFilePath = "logs/facialrecognition.log")
    {
        Instance = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.Console(
                formatProvider: CultureInfo.InvariantCulture,
                outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .WriteTo.File(
                logFilePath,
                formatProvider: CultureInfo.InvariantCulture,
                rollingInterval: RollingInterval.Day,
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        Log.Logger = Instance;
        Instance.Information("Logger initialized. Output: {Path}", logFilePath);
    }

    /// <summary>
    /// Orchestrates the safe termination of the logging subsystem.
    /// Flushes any pending messages in the async buffers to stable storage.
    /// </summary>
    public static void CloseAndFlush() => Log.CloseAndFlush();

}
