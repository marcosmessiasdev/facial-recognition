using System;
using System.Runtime.InteropServices;
using Windows.Graphics.DirectX.Direct3D11;

namespace WindowCapture
{
    /// <summary>
    /// Utility class to assist with Direct3D11 device creation and WinRT interop.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Bridges the gap between native DirectX 11 APIs and the Modern Windows Runtime (WinRT) 
    /// interfaces required for Graphics Capture.
    ///
    /// Responsibilities:
    /// - Execute native P/Invoke calls to create D3D11 hardware devices.
    /// - Convert native ID3D11Device pointers into WinRT-compatible IDirect3DDevice objects.
    /// - Handle HRESULT error validation for DirectX initialization.
    ///
    /// Dependencies:
    /// - D3D11.dll (Native DirectX library)
    /// - WinRT.Runtime (DirectX-WinRT marshaling)
    ///
    /// Architectural Role:
    /// Infrastructure Component / Helper.
    ///
    /// Constraints:
    /// - Primarily used during the initialization phase of GraphicsCaptureService.
    /// </remarks>
    public static class Direct3D11Helper
    {
        [DllImport("d3d11.dll", EntryPoint = "D3D11CreateDevice", SetLastError = true)]
        private static extern int D3D11CreateDevice(
            IntPtr pAdapter,
            int driverType,
            IntPtr Software,
            uint Flags,
            IntPtr pFeatureLevels,
            uint FeatureLevels,
            uint SDKVersion,
            out IntPtr ppDevice,
            out int pFeatureLevel,
            out IntPtr ppImmediateContext);

        [DllImport("d3d11.dll", EntryPoint = "CreateDirect3D11DeviceFromDXGIDevice", SetLastError = true)]
        private static extern int CreateDirect3D11DeviceFromDXGIDevice(IntPtr dxgiDevice, out IntPtr graphicsDevice);

        /// <summary>
        /// Creates a new Direct3D11 device with BGRA support and returns it as a WinRT IDirect3DDevice.
        /// </summary>
        /// <returns>An initialized IDirect3DDevice instance.</returns>
        /// <exception cref="Exception">Thrown if the device creation or interface query fails.</exception>
        public static IDirect3DDevice CreateDevice()
        {
            // D3D_DRIVER_TYPE_HARDWARE = 1
            // D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
            // D3D11_SDK_VERSION = 7
            
            int hr = D3D11CreateDevice(
                IntPtr.Zero,
                1,
                IntPtr.Zero,
                0x20,
                IntPtr.Zero,
                0,
                7,
                out IntPtr d3dDevice,
                out _,
                out _);

            if (hr != 0)
            {
                throw new Exception($"Failed to create D3D11 device. Setup D3D11 CreateDevice failed with HRESULT 0x{hr:X8}");
            }

            var dxgiDeviceGuid = new Guid("54ec77fa-1377-44e6-8c32-88fd5f44c84c"); // IDXGIDevice
            hr = Marshal.QueryInterface(d3dDevice, ref dxgiDeviceGuid, out IntPtr dxgiDevice);
            
            if (hr != 0)
            {
                Marshal.Release(d3dDevice);
                throw new Exception($"Failed to query IDXGIDevice. HRESULT 0x{hr:X8}");
            }

            hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, out IntPtr inspectableDevice);

            Marshal.Release(dxgiDevice);
            Marshal.Release(d3dDevice);
            
            if (hr != 0 || inspectableDevice == IntPtr.Zero)
            {
                throw new Exception($"Failed to create WinRT Direct3D11 device from DXGI device. HRESULT 0x{hr:X8}");
            }

            var device = WinRT.MarshalInspectable<IDirect3DDevice>.FromAbi(inspectableDevice);
            Marshal.Release(inspectableDevice);
            return device;
        }
    }
}

