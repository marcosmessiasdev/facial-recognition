# Domain Model

## Entities

### `MeetingSession`
The top-level root. Represents one full recording session.
- **StartedAtUtc**: The time `VisionEngine.Start()` was called.
- **EndedAtUtc**: The time it was stopped.
- **Transcript**: Full list of transcribed sentences with speaker identifiers.
- **Segments**: `SpeakerSegment` items marking exact start/stop intervals for every spoken event.
- **Aggregates**: Statistical summaries (e.g., `DominantSpeakers`).

### `Track` (from `FaceTracking`)
An ephemeral memory construct matching bounding boxes across contiguous video frames. If a person drops off-screen and returns, they are assigned a new Track ID, and their ArcFace embedding maps them back to the same logical `Person`.

### `Person` (from `IdentityStore`)
A confirmed physical identity.
- **Id**: Guid acting as the database PK.
- **Name**: User-assigned name representing the person.
- **Embedding**: An array of floating-point numbers encoding unique facial structures from ArcFace.
- **ImageBase64**: The high-res visual crop snapshot.

### `VisionFrame`
An atomic encapsulation linking a timestamp with raw bytes. Holds the OpenCvSharp `Mat` and `SampleRate` audio pointers aligned.

### `AudioSpeakerSegment`
Used in the `SpeakerDiarization` module to bucketize consecutive audio bytes containing human speech. Later clustered using aggressive cosine constraints mapping it to a real `PersonName`.

## Concept Flow
1. Raw `VisionFrame` instances feed into detection.
2. The detected face becomes a `Track`.
3. If the face matches a known ArcFace array, the `Track` acquires a `PersonName` reference pointing to `Person`.
4. When `TalkNetAsdModel` or `SileroVad` detects speech while the face's mouth is moving, a `SpeakerSegment` is created for that `Person`.
5. When the user clicks Stop, all uncommitted data commits into `MeetingSession`, which serializes the output to disk.
