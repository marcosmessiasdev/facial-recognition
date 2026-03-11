using System;

namespace SpeechProcessing;

public sealed record TranscriptSegment(TimeSpan Start, TimeSpan End, string Text);

