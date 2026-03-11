using System;

namespace AudioProcessing;

/// <summary>
/// Provides data for the FrameArrived event of the LoopbackAudioCapture class.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Encapsulates a single chunk of processed audio data along with its metadata.
///
/// Responsibilities:
/// - Transport raw float samples from the capture source to subscribers.
/// - Carry sample rate and timing information for synchronization.
///
/// Architectural Role:
/// Data Transfer Object (DTO) / Event Arguments.
/// </remarks>
public sealed class AudioFrameEventArgs : EventArgs
{
    /// <summary>
    /// Initializes a new instance of the AudioFrameEventArgs class.
    /// </summary>
    /// <param name="samples">The mono audio samples.</param>
    /// <param name="sampleRateHz">The sample rate in Hertz.</param>
    /// <param name="offset">The monotonic time offset.</param>
    public AudioFrameEventArgs(float[] samples, int sampleRateHz, TimeSpan offset)
    {
        Samples = samples;
        SampleRateHz = sampleRateHz;
        Offset = offset;
    }

    /// <summary>
    /// Gets the mono audio samples in range [-1..1].
    /// </summary>
    public float[] Samples { get; }

    /// <summary>
    /// Gets the sample rate of the audio in Hertz.
    /// </summary>
    public int SampleRateHz { get; }

    /// <summary>
    /// Gets the time offset since capture started (monotonic).
    /// </summary>
    public TimeSpan Offset { get; }
}


