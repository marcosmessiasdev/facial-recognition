namespace SpeakerDiarization;

public sealed record AudioSpeakerSegment(TimeSpan Start, TimeSpan End, int SpeakerId, float Confidence);

