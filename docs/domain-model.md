# Domain Model

The system operates on a set of core entities that represent the state of the physical world being monitored.

## Key Entities

### VisionFrame
- **Role**: Unit of Work.
- **Responsibility**: Holds the raw pixels and spatial metadata of a single moment in time.
- **Lifecycle**: Temporary, usually disposed after one pipeline pass.

### Track
- **Role**: State Container.
- **Responsibility**: Represents a persisting human identity in the video stream.
- **Attributes**: Stable ID, Bounding Box, Name, Emotion, Age, Gender, Mouth Motion Score, Speaking Status.

### Person
- **Role**: Persistent Identity.
- **Responsibility**: Represents a registered user in the permanent database.
- **Attributes**: Name, Facial Embedding (512-float vector).

### MeetingSession
- **Role**: Aggregate Root for Analytics.
- **Responsibility**: Encapsulates a start-to-finish meeting period.
- **Content**: Chronicle of `SpeakerSegments` and calculated summary statistics.

### SpeakerSegment
- **Role**: Value Object / Event Record.
- **Responsibility**: Records a period where a specific identifier was identified as the active speaker.

## Relationships
- A **Track** can be associated with one **Person** via embedding similarity.
- A **MeetingSession** contains many **SpeakerSegments**.
- **SpeakerSegments** reference either a **Track ID** (anonymous) or a **Person Name** (identified).
