namespace AudioProcessing;

public interface IAudioCapture : IDisposable
{
    event EventHandler<AudioFrameEventArgs>? FrameArrived;
    int TargetSampleRateHz { get; }
    int FrameSizeSamples { get; }
    string? SelectedDeviceName { get; }
    void Start();
    void StopCapture();
}
