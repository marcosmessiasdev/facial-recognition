using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Controls;
using FaceTracking;
using FaceDetection;

namespace OverlayRenderer;

/// <summary>
/// A transparent WPF window that renders bounding boxes and metadata as an overlay on top of other windows.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Provides the visual feedback layer of the system, displaying where faces are detected and 
/// showing their respective attributes (name, emotion, etc.) in real-time.
///
/// Responsibilities:
/// - Maintain a transparent, top-most window that allows mouse click-through using Win32 extended styles.
/// - Dynamically position and size itself to match the target window being captured.
/// - Draw vector graphics (Rectangles and TextBlocks) for each active track.
///
/// Dependencies:
/// - WPF (Windowing and rendering)
/// - FaceTracking (Track data source)
/// - User32.dll (For click-through capability)
///
/// Architectural Role:
/// Presentation Layer / View.
///
/// Constraints:
/// - Must be run on the UI (STA) thread.
/// - Only renders labels; does not own any business logic.
/// </remarks>
public partial class BoundingBoxOverlay : Window
{
    private readonly Canvas _canvas;

    /// <summary>
    /// Initializes a new instance of the BoundingBoxOverlay class with transparency and click-through enabled.
    /// </summary>
    public BoundingBoxOverlay()
    {
        WindowStyle = WindowStyle.None;
        AllowsTransparency = true;
        Background = Brushes.Transparent;
        Topmost = true;
        ShowInTaskbar = false;

        _canvas = new Canvas { Background = Brushes.Transparent };
        Content = _canvas;
    }

    /// <summary>
    /// Sets the window to be transparent to mouse clicks upon initialization.
    /// </summary>
    /// <param name="e">Event arguments.</param>
    protected override void OnSourceInitialized(EventArgs e)
    {
        base.OnSourceInitialized(e);
        nint hwnd = new WindowInteropHelper(this).Handle;
        int exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
        _ = SetWindowLong(hwnd, GWL_EXSTYLE, exStyle | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW);
    }

    /// <summary>
    /// Synchronizes the overlay window with the target window state and renders active tracks.
    /// </summary>
    /// <param name="tracks">
    /// The list of active face tracks containing position and classification metadata.
    /// </param>
    /// <param name="windowWidth">The current physical width of the window being monitored.</param>
    /// <param name="windowHeight">The current physical height of the window being monitored.</param>
    /// <param name="overlayLeft">The target X-coordinate for the overlay window on the desktop.</param>
    /// <param name="overlayTop">The target Y-coordinate for the overlay window on the desktop.</param>
    /// <remarks>
    /// This method performs two primary tasks:
    /// 1. Resizes and re-positions the WPF Window to exactly match the target application's boundaries.
    /// 2. Clears and rebuilds the visual Canvas with rectangles and text labels based on updated tracking data.
    /// </remarks>
    public void UpdateTracks(List<Track> tracks, int windowWidth, int windowHeight, int overlayLeft, int overlayTop, string? hudText = null)
    {
        ArgumentNullException.ThrowIfNull(tracks);

        Left = overlayLeft;
        Top = overlayTop;
        Width = windowWidth;
        Height = windowHeight;

        _canvas.Children.Clear();

        if (!string.IsNullOrWhiteSpace(hudText))
        {
            TextBlock hud = new()
            {
                Text = hudText,
                Foreground = Brushes.White,
                Background = new SolidColorBrush(Color.FromArgb(180, 0, 0, 0)),
                FontSize = 12,
                Padding = new Thickness(6, 4, 6, 4),
                TextWrapping = TextWrapping.Wrap,
                MaxWidth = Math.Max(200, windowWidth * 0.45)
            };
            Canvas.SetLeft(hud, 8);
            Canvas.SetTop(hud, 8);
            _ = _canvas.Children.Add(hud);
        }

        foreach (Track track in tracks)
        {
            BoundingBox box = track.Box;

            // Bounding box rectangle
            Rectangle rect = new()
            {
                Stroke = track.IsSpeaking ? Brushes.Yellow : Brushes.LimeGreen,
                StrokeThickness = 2.5,
                Width = box.Width,
                Height = box.Height
            };
            Canvas.SetLeft(rect, box.X);
            Canvas.SetTop(rect, box.Y);
            _ = _canvas.Children.Add(rect);

            // Label: Name + Emotion
            string label = "";
            if (!string.IsNullOrEmpty(track.PersonName))
            {
                label += track.PersonName;
            }

            if (track.IsSpeaking)
            {
                label += "  [Speaking]";
            }

            if (track.SpeakingScore > 0.0001f)
            {
                label += $"  [SpeakScore {track.SpeakingScore:P0}]";
            }

            if (track.TalkNetSpeakingProb > 0.0001f)
            {
                label += $"  [TalkNet {track.TalkNetSpeakingProb:P0}]";
            }

            if (!string.IsNullOrEmpty(track.EmotionLabel))
            {
                label += $"  [{track.EmotionLabel}]";
            }

            if (!string.IsNullOrEmpty(track.GenderLabel))
            {
                label += $"  [{track.GenderLabel}]";
            }

            if (!string.IsNullOrEmpty(track.AgeLabel))
            {
                label += $"  [{track.AgeLabel}]";
            }

            if (string.IsNullOrEmpty(label))
            {
                label = $"#{track.Id}  {box.Confidence:P0}";
            }

            TextBlock text = new()
            {
                Text = label,
                Foreground = track.IsSpeaking ? Brushes.Yellow : Brushes.LimeGreen,
                Background = new SolidColorBrush(Color.FromArgb(160, 0, 0, 0)),
                FontSize = 12,
                Padding = new Thickness(4, 2, 4, 2)
            };
            Canvas.SetLeft(text, box.X);
            Canvas.SetTop(text, Math.Max(0, box.Y - 22));
            _ = _canvas.Children.Add(text);
        }
    }

    private const int GWL_EXSTYLE = -20;
    private const int WS_EX_TRANSPARENT = 0x00000020;
    private const int WS_EX_LAYERED = 0x00080000;
    private const int WS_EX_TOOLWINDOW = 0x00000080;

    /// <summary>Retrieves information about the specified window using Win32 API.</summary>
    [LibraryImport("user32.dll", EntryPoint = "GetWindowLongW", SetLastError = true)]
    private static partial int GetWindowLong(IntPtr hwnd, int index);

    /// <summary>Changes an attribute of the specified window using Win32 API.</summary>
    [LibraryImport("user32.dll", EntryPoint = "SetWindowLongW", SetLastError = true)]
    private static partial int SetWindowLong(IntPtr hwnd, int index, int newStyle);

}
