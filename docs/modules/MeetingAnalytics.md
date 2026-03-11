# Module: MeetingAnalytics

## Purpose
Collects and aggregates real-time speaker events into meaningful participation metrics.

## Key Components
- **MeetingAnalyticsEngine**: The state machine for event collection.
- **MeetingSession**: Data container for a finished meeting.
- **SpeakerSegment**: Record of a specific person's speaking interval.

## Responsibilities
- Record start and end timestamps for every "Active Speaker" transition.
- Aggregate total duration per participant (Identified or Track ID).
- Detect interruptions based on rapid speaker transitions (< 500ms).
- Serialize results to highly readable JSON formats for post-meeting analysis.

## Dependencies
- **FaceTracking**: Provides names and IDs of speakers.
- **SpeakerDetection**: Signals the source of analytics events.

## Output Format
Sessions are saved to the `logs/` directory with timestamps, e.g., `meeting_session_20231027_153000.json`.
