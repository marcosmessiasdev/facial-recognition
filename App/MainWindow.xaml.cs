using System.Runtime.InteropServices;
using System.Windows;
using System.Diagnostics;
using System.IO;
using VisionEngine;

namespace FacialRecognitionApp;

/// <summary>
/// Helper class for window enumeration.
/// </summary>
public class WindowInfo
{
    /// <summary>The window title.</summary>
    public string Title { get; set; } = "";
    /// <summary>The native handle (HWND) of the window.</summary>
    public IntPtr Handle { get; set; }
    /// <summary>Returns the title for display purposes.</summary>
    public override string ToString() => Title;
}

/// <summary>
/// Main controller and user interface for the facial recognition application.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Provides a visual interface for selecting a target window and managing the 
/// facial recognition pipeline lifecycle.
///
/// Responsibilities:
/// - Enumerate active desktop windows for capture selection.
/// - Orchestrate the startup and shutdown of the VisionPipeline.
/// - Display real-time status updates and handle user interaction events.
///
/// Dependencies:
/// - VisionEngine (VisionPipeline core)
/// - User32.dll (Win32 Interop for window enumeration)
///
/// Architectural Role:
/// Application Controller / View.
/// </remarks>
public partial class MainWindow : Window, IDisposable
{
    private VisionPipeline? _visionPipeline;
    private IntPtr _selectedHwnd;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the MainWindow.
    /// </summary>
    public MainWindow()
    {
        InitializeComponent();
        Loaded += (_, _) => LoadOpenWindows();
    }

    /// <summary>
    /// Populates the window list with currently open and visible top-level windows.
    /// </summary>
    private void LoadOpenWindows()
    {
        List<WindowInfo> windows = new();

        _ = EnumWindows((hwnd, _) =>
        {
            if (!IsWindowVisible(hwnd))
            {
                return true;
            }

            string title = GetWindowTitle(hwnd);
            if (!string.IsNullOrWhiteSpace(title))
            {
                windows.Add(new WindowInfo { Handle = hwnd, Title = title });
            }

            return true;
        }, IntPtr.Zero);

        WindowListBox.ItemsSource = windows;
    }

    private static string GetWindowTitle(IntPtr hwnd)
    {
        int len = GetWindowTextLength(hwnd);
        if (len <= 0)
        {
            return "";
        }

        char[] buffer = new char[len + 1];
        int copied = GetWindowText(hwnd, buffer, buffer.Length);
        return copied <= 0 ? "" : new string(buffer, 0, copied).Trim();
    }

    /// <summary>
    /// Refreshes the list of available windows.
    /// </summary>
    private void RefreshButton_Click(object sender, RoutedEventArgs e)
    {
        LoadOpenWindows();
        StatusText.Text = "Lista atualizada.";
    }

    private void DemoButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            string basedir = AppDomain.CurrentDomain.BaseDirectory;
            string demo = Path.Combine(basedir, "demo", "index.html");
            if (!File.Exists(demo))
            {
                StatusText.Text = "Demo não encontrado (demo/index.html).";
                return;
            }

            string url = new Uri(demo).AbsoluteUri;
            _ = Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            StatusText.Text = "Demo aberto no navegador.";
        }
        catch (Exception ex)
        {
            StatusText.Text = "Falha ao abrir demo: " + ex.Message;
        }
    }

    /// <summary>
    /// Handles the selection change event in the window list.
    /// </summary>
    private void WindowListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (WindowListBox.SelectedItem is WindowInfo info)
        {
            _selectedHwnd = info.Handle;
            StatusText.Text = $"Selecionado: {info.Title}";
            StartButton.IsEnabled = true;
        }
    }

    /// <summary>
    /// Initializes and starts the vision pipeline for the selected window.
    /// </summary>
    private void StartButton_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedHwnd == IntPtr.Zero)
        {
            return;
        }

        _visionPipeline?.Dispose();
        _visionPipeline = new VisionPipeline();

        _visionPipeline.Initialize(); // paths come from appsettings.json via AppConfig
        _visionPipeline.Start(_selectedHwnd);

        StartButton.Visibility = Visibility.Collapsed;
        StopButton.Visibility = Visibility.Visible;
        StatusText.Text = "Analisando...";
    }

    /// <summary>
    /// Stops and disposes the current vision pipeline.
    /// </summary>
    private void StopButton_Click(object sender, RoutedEventArgs e)
    {
        _visionPipeline?.Dispose();
        _visionPipeline = null;

        StartButton.Visibility = Visibility.Visible;
        StopButton.Visibility = Visibility.Collapsed;
        StatusText.Text = "Parado.";
    }

    /// <summary>
    /// Ensures resources are released when the window is closed.
    /// </summary>
    protected override void OnClosed(EventArgs e)
    {
        Dispose();
        base.OnClosed(e);
    }

    /// <summary>
    /// Disposes owned resources (VisionPipeline).
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _visionPipeline?.Dispose();
        _visionPipeline = null;
        GC.SuppressFinalize(this);
    }

    // ─── Win32 Interop ───────────────────────────────────────────────────

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", EntryPoint = "GetWindowTextW", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern int GetWindowText(IntPtr hWnd, [Out] char[] lpString, int nMaxCount);

    [DllImport("user32.dll", EntryPoint = "GetWindowTextLengthW", SetLastError = true)]
    private static extern int GetWindowTextLength(IntPtr hWnd);
}
