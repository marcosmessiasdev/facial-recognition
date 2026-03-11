using System.Runtime.InteropServices;
using Windows.Graphics.Capture;
using Windows.Graphics.DirectX;
using Windows.Graphics.DirectX.Direct3D11;
using Vortice.Direct3D11;
using Logging;

namespace WindowCapture;

/// <summary>
/// Service responsible for capturing video frames from a specific window using the Windows Graphics Capture API.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Provides a high-performance mechanism to capture the contents of a target application window 
/// and deliver raw pixel data to the vision pipeline.
///
/// Responsibilities:
/// - Initialize and manage the Windows Graphics Capture session for a given HWND.
/// - Orchestrate the Direct3D11 frame pool and handle frame arrival events.
/// - Perform GPU-to-CPU memory copies using staging textures to extract raw BGRA pixel data.
/// - Handle dynamic window resizing by recreating the frame pool as needed.
/// - Provide thread-safe resource management for the underlying DirectX infrastructure.
///
/// Dependencies:
/// - Windows.Graphics.Capture (System-level capture API)
/// - Vortice.Direct3D11 (DirectX wrapper for .NET)
/// - Direct3D11Helper (Interop utilities)
///
/// Architectural Role:
/// Infrastructure Component / Data Source.
///
/// Constraints:
/// - Requires Windows 10 version 1903 (18362) or later for Graphics Capture.
/// - Heavy lifting is performed in the frame pool thread; avoid blocking in the FrameArrived event.
/// </remarks>
public sealed partial class GraphicsCaptureService : IDisposable
{
    private static readonly Guid IID_ID3D11Device = new("db6f6ddb-ac77-4e88-8253-819df9bbf140");
    private static readonly Guid IID_ID3D11Texture2D = new("6f15aaf2-d208-4e89-9ab4-489535d34f9c");
    private static readonly Guid IID_IDirect3DDxgiInterfaceAccess = new("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1");

    private readonly object _sync = new();

    private GraphicsCaptureItem? _captureItem;
    private Direct3D11CaptureFramePool? _framePool;
    private GraphicsCaptureSession? _session;
    private IDirect3DDevice? _device;

    private ID3D11Device? _vorticeDevice;
    private ID3D11DeviceContext? _vorticeContext;

    private byte[]? _frameBuffer;
    private int _lastWidth;
    private int _lastHeight;
    private bool _isDisposed;

    /// <summary>
    /// Occurs when a new raw video frame has been captured and is ready for processing.
    /// </summary>
    public event EventHandler<(byte[] data, int width, int height, int stride)>? RawFrameArrived;

    /// <summary>
    /// Starts the high-performance capture process for the specified window handle.
    /// </summary>
    /// <param name="hwnd">The handle of the window to capture.</param>
    /// <exception cref="ArgumentException">Thrown when the window handle is invalid.</exception>
    public void StartCapture(IntPtr hwnd)
    {
        if (hwnd == IntPtr.Zero)
        {
            throw new ArgumentException("Window handle cannot be zero.", nameof(hwnd));
        }

        lock (_sync)
        {
            ThrowIfDisposed();

            StopCaptureInternal();

            _device = Direct3D11Helper.CreateDevice();

            IntPtr d3dDevicePtr = GetDXGIPointer(_device, IID_ID3D11Device);
            if (d3dDevicePtr == IntPtr.Zero)
            {
                throw new InvalidOperationException("Unable to obtain ID3D11Device pointer from WinRT device.");
            }

            try
            {
                _vorticeDevice = new ID3D11Device(d3dDevicePtr);
                _vorticeContext = _vorticeDevice.ImmediateContext;
                d3dDevicePtr = IntPtr.Zero; // ownership transferred to the wrapper
            }
            finally
            {
                if (d3dDevicePtr != IntPtr.Zero)
                {
                    _ = Marshal.Release(d3dDevicePtr);
                }
            }

            _captureItem = CreateItemForWindow(hwnd)
                ?? throw new InvalidOperationException("Unable to create GraphicsCaptureItem for the selected window.");

            _captureItem.Closed += OnCaptureItemClosed;

            _lastWidth = _captureItem.Size.Width;
            _lastHeight = _captureItem.Size.Height;

            _framePool = Direct3D11CaptureFramePool.CreateFreeThreaded(
                _device,
                DirectXPixelFormat.B8G8R8A8UIntNormalized,
                2,
                _captureItem.Size);

            _framePool.FrameArrived += OnFrameArrived;

            _session = _framePool.CreateCaptureSession(_captureItem);
            _session.StartCapture();
        }
    }

    /// <summary>
    /// Stops the current capture session and releases associated resources.
    /// </summary>
    public void StopCapture()
    {
        lock (_sync)
        {
            StopCaptureInternal();
        }
    }

    /// <summary>
    /// Internal method to stop capture and cleanup resources.
    /// </summary>
    private void StopCaptureInternal()
    {
        GraphicsCaptureSession? session = _session;
        Direct3D11CaptureFramePool? pool = _framePool;
        GraphicsCaptureItem? item = _captureItem;

        _session = null;
        _framePool = null;
        _captureItem = null;

        try
        {
            if (pool != null)
            {
                pool.FrameArrived -= OnFrameArrived;
            }
        }
        catch { /* ignore */ }

        try { session?.Dispose(); } catch { /* ignore */ }
        try { pool?.Dispose(); } catch { /* ignore */ }

        try
        {
            if (item != null)
            {
                item.Closed -= OnCaptureItemClosed;
            }
        }
        catch { /* ignore */ }

        _vorticeContext?.Dispose();
        _vorticeContext = null;

        _vorticeDevice?.Dispose();
        _vorticeDevice = null;

        _device?.Dispose();
        _device = null;

        _frameBuffer = null;
        _lastWidth = 0;
        _lastHeight = 0;
    }

    /// <summary>
    /// Callback for when the captured item (window) is closed.
    /// </summary>
    private void OnCaptureItemClosed(GraphicsCaptureItem sender, object args)
    {
        StopCapture();
    }

    /// <summary>
    /// Callback for when a new frame is available in the capture pool.
    /// </summary>
    private void OnFrameArrived(Direct3D11CaptureFramePool sender, object args)
    {
        try
        {
            using Direct3D11CaptureFrame frame = sender.TryGetNextFrame();
            if (frame == null)
            {
                return;
            }

            ID3D11Device? vorticeDevice = _vorticeDevice;
            ID3D11DeviceContext? vorticeContext = _vorticeContext;
            IDirect3DDevice? device = _device;
            if (vorticeDevice == null || vorticeContext == null || device == null)
            {
                return;
            }

            // 1. Handle resize/fullscreen
            if (_lastWidth != 0 && _lastHeight != 0 &&
                (frame.ContentSize.Width != _lastWidth || frame.ContentSize.Height != _lastHeight))
            {
                _lastWidth = frame.ContentSize.Width;
                _lastHeight = frame.ContentSize.Height;
                sender.Recreate(
                    device,
                    DirectXPixelFormat.B8G8R8A8UIntNormalized,
                    2,
                    frame.ContentSize);
                return;
            }

            _lastWidth = frame.ContentSize.Width;
            _lastHeight = frame.ContentSize.Height;

            IDirect3DSurface surface = frame.Surface;
            // Note: We don't manually dispose 'surface' here because 'frame' owns it
            // and will release it when disposed. Manual dispose can cause race conditions.

            IntPtr texturePtr = GetDXGIPointer(surface, IID_ID3D11Texture2D);
            if (texturePtr == IntPtr.Zero)
            {
                return;
            }

            try
            {
                using ID3D11Texture2D texture = new(texturePtr);
                texturePtr = IntPtr.Zero; // ownership transferred to the wrapper

                Texture2DDescription desc = texture.Description;
                desc.Usage = ResourceUsage.Staging;
                desc.BindFlags = BindFlags.None;
                desc.CPUAccessFlags = CpuAccessFlags.Read;
                desc.MiscFlags = ResourceOptionFlags.None;

                using ID3D11Texture2D stagingTexture = vorticeDevice.CreateTexture2D(desc);
                vorticeContext.CopyResource(stagingTexture, texture);

                MappedSubresource mappedResource = vorticeContext.Map(stagingTexture, 0, MapMode.Read, MapFlags.None);
                try
                {
                    int width = (int)desc.Width;
                    int height = (int)desc.Height;
                    int stride = (int)mappedResource.RowPitch;
                    int rowBytes = width * 4; // BGRA8 = 4 bytes per pixel

                    // 2. Reusable buffer to avoid GC pressure
                    int totalSize = rowBytes * height;
                    byte[]? buffer = _frameBuffer;
                    if (buffer == null || buffer.Length != totalSize)
                    {
                        buffer = new byte[totalSize];
                        _frameBuffer = buffer;
                    }

                    // 3. Robust copy: row by row to skip GPU padding (stride vs width*4)
                    for (int y = 0; y < height; y++)
                    {
                        IntPtr sourcePtr = IntPtr.Add(mappedResource.DataPointer, y * stride);
                        Marshal.Copy(sourcePtr, buffer, y * rowBytes, rowBytes);
                    }

                    // Clone for thread safety in subscribers
                    byte[] clonedBuffer = new byte[totalSize];
                    Buffer.BlockCopy(buffer, 0, clonedBuffer, 0, totalSize);

                    RawFrameArrived?.Invoke(this, (clonedBuffer, width, height, rowBytes));
                }
                finally
                {
                    try
                    {
                        vorticeContext.Unmap(stagingTexture, 0);
                    }
                    catch
                    {
                        // If capture is being stopped concurrently, the D3D context may already be disposed.
                    }
                }
            }
            finally
            {
                if (texturePtr != IntPtr.Zero)
                {
                    _ = Marshal.Release(texturePtr);
                }
            }
        }
        catch (Exception ex)
        {
            AppLogger.Instance.Error(ex, "GraphicsCaptureService.OnFrameArrived failed");
        }
    }

    /// <summary>
    /// Ensures the frame buffer is initialized and of the correct size.
    /// </summary>
    private void EnsureFrameBuffer(int size)
    {
        if (_frameBuffer == null || _frameBuffer.Length != size)
        {
            _frameBuffer = new byte[size];
        }
    }

    /// <summary>
    /// Throws an ObjectDisposedException if the service has been disposed.
    /// </summary>
    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_isDisposed, this);
    }

    /// <summary>
    /// Releases all resources used by the capture service.
    /// </summary>
    public void Dispose()
    {
        if (_isDisposed)
        {
            return;
        }

        lock (_sync)
        {
            if (_isDisposed)
            {
                return;
            }

            StopCaptureInternal();
            _isDisposed = true;
        }

        GC.SuppressFinalize(this);
    }

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int GetInterfaceDelegate(IntPtr thisPtr, ref Guid iid, out IntPtr p);

    /// <summary>
    /// Helper to retrieve the underlying DXGI pointer for a WinRT object.
    /// </summary>
    private static IntPtr GetDXGIPointer(object winrtObject, Guid targetIid)
    {
        bool releaseUnk = false;

        IntPtr unk;
        if (winrtObject is WinRT.IWinRTObject winrtObj)
        {
            unk = winrtObj.NativeObject.ThisPtr;
        }
        else
        {
            unk = Marshal.GetIUnknownForObject(winrtObject);
            releaseUnk = true;
        }

        if (unk == IntPtr.Zero)
        {
            return IntPtr.Zero;
        }

        Guid dxgiAccessIid = IID_IDirect3DDxgiInterfaceAccess;
        int hr = Marshal.QueryInterface(unk, ref dxgiAccessIid, out IntPtr accessPtr);

        if (releaseUnk)
        {
            _ = Marshal.Release(unk);
        }

        if (hr != 0 || accessPtr == IntPtr.Zero)
        {
            return IntPtr.Zero;
        }

        try
        {
            IntPtr vtable = Marshal.ReadIntPtr(accessPtr);
            IntPtr methodPtr = Marshal.ReadIntPtr(vtable, 3 * IntPtr.Size);
            GetInterfaceDelegate getInterface = Marshal.GetDelegateForFunctionPointer<GetInterfaceDelegate>(methodPtr);

            hr = getInterface(accessPtr, ref targetIid, out IntPtr resultPtr);
            return hr == 0 ? resultPtr : IntPtr.Zero;
        }
        finally
        {
            _ = Marshal.Release(accessPtr);
        }
    }

    [LibraryImport("combase.dll", EntryPoint = "WindowsCreateString", StringMarshalling = StringMarshalling.Utf16)]
    private static partial int WindowsCreateString(string sourceString, int length, out IntPtr hstring);

    [LibraryImport("combase.dll", EntryPoint = "WindowsDeleteString")]
    private static partial int WindowsDeleteString(IntPtr hstring);

    [LibraryImport("combase.dll", EntryPoint = "RoGetActivationFactory")]
    private static partial int RoGetActivationFactory(IntPtr activatableClassId, ref Guid iid, out IntPtr factory);

    /// <summary>
    /// WinRT helper to create a GraphicsCaptureItem for an HWND.
    /// </summary>
    private static GraphicsCaptureItem? CreateItemForWindow(IntPtr hwnd)
    {
        IntPtr hString = IntPtr.Zero;
        IntPtr factoryPtr = IntPtr.Zero;
        IntPtr rawItemPtr = IntPtr.Zero;

        try
        {
            Guid iidInterop = new("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356");
            string runtimeClassId = "Windows.Graphics.Capture.GraphicsCaptureItem";

            int hr = WindowsCreateString(runtimeClassId, runtimeClassId.Length, out hString);
            if (hr != 0 || hString == IntPtr.Zero)
            {
                return null;
            }

            hr = RoGetActivationFactory(hString, ref iidInterop, out factoryPtr);
            if (hr != 0 || factoryPtr == IntPtr.Zero)
            {
                return null;
            }

            IGraphicsCaptureItemInterop factory = (IGraphicsCaptureItemInterop)Marshal.GetObjectForIUnknown(factoryPtr);

            Guid iidCaptureItem = new("79C3F95B-31F7-4EC2-A464-632EF5D30760");
            rawItemPtr = factory.CreateForWindow(hwnd, ref iidCaptureItem);
            if (rawItemPtr == IntPtr.Zero)
            {
                return null;
            }

            GraphicsCaptureItem captureItem = GraphicsCaptureItem.FromAbi(rawItemPtr);
            return captureItem;
        }
        catch
        {
            return null;
        }
        finally
        {
            if (rawItemPtr != IntPtr.Zero)
            {
                _ = Marshal.Release(rawItemPtr);
            }

            if (factoryPtr != IntPtr.Zero)
            {
                _ = Marshal.Release(factoryPtr);
            }

            if (hString != IntPtr.Zero)
            {
                _ = WindowsDeleteString(hString);
            }
        }
    }
}

/// <summary>
/// Native interface for specialized GraphicsCaptureItem creation from a window or monitor handle.
/// </summary>
[ComImport]
[Guid("3628E81B-3CAC-4C60-B7F4-23CE0E0C3356")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IGraphicsCaptureItemInterop
{
    IntPtr CreateForWindow(
        [In] IntPtr window,
        [In] ref Guid iid);

    IntPtr CreateForMonitor(
        [In] IntPtr monitor,
        [In] ref Guid iid);
}
