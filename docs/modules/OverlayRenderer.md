# Module: OverlayRenderer

## Purpose
Provides visual feedback by drawing metadata directly on top of the monitored application.

## Key Components
- **BoundingBoxOverlay**: A transparent, topmost WPF window.

## Responsibilities
- Synchronize window position and size with the target application window using Win32 `GetWindowRect`.
- Use Win32 Extended Styles (`WS_EX_TRANSPARENT`) to ensure mouse clicks pass through the overlay to the underlying app.
- Render hardware-accelerated vector graphics (Rectangles, TextBlocks) to represent face tracks.
- Use color coding (e.g., Yellow for speaking, Green for idle) to convey status.

## Dependencies
- **WPF (System.Windows)**: For rendering and layout.
- **User32.dll**: For low-level window style manipulation.

## Architectural Role
**Presentation Layer**: The primary output interface for the end-user.
