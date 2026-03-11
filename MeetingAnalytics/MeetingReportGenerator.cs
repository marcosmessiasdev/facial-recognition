using System.Globalization;
using System.Text;

namespace MeetingAnalytics;

public static class MeetingReportGenerator
{
    public static string PersistHtml(MeetingSession session, string baseDir)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDir);

        _ = Directory.CreateDirectory(Path.Combine(baseDir, "logs"));
        string path = Path.Combine(baseDir, "logs", $"meeting_report_{session.StartedAtUtc:yyyyMMdd_HHmmss}.html");
        File.WriteAllText(path, BuildHtml(session), Encoding.UTF8);
        return path;
    }

    public static string PersistTranscript(MeetingSession session, string baseDir)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDir);

        _ = Directory.CreateDirectory(Path.Combine(baseDir, "logs"));
        string path = Path.Combine(baseDir, "logs", $"meeting_transcript_{session.StartedAtUtc:yyyyMMdd_HHmmss}.txt");

        StringBuilder sb = new();
        foreach (Utterance u in session.Utterances.OrderBy(u => u.StartUtc))
        {
            string who = u.DisplayName ?? u.SpeakerKey;
            sb.Append('[').Append(u.StartUtc.ToLocalTime().ToString("HH:mm:ss", CultureInfo.InvariantCulture)).Append("] ");
            sb.Append(who).Append(": ").AppendLine(u.Text);
        }

        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
        return path;
    }

    public static string PersistSpeakerTimelineCsv(MeetingSession session, string baseDir)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDir);

        _ = Directory.CreateDirectory(Path.Combine(baseDir, "logs"));
        string path = Path.Combine(baseDir, "logs", $"meeting_timeline_{session.StartedAtUtc:yyyyMMdd_HHmmss}.csv");

        StringBuilder sb = new();
        sb.AppendLine("start_utc,end_utc,speaker_key,display_name,track_id,audio_speaker_id");
        foreach (SpeakerSegment s in session.Segments.OrderBy(s => s.StartUtc))
        {
            sb.Append(s.StartUtc.ToString("O", CultureInfo.InvariantCulture)).Append(',');
            sb.Append((s.EndUtc ?? session.EndedAtUtc ?? DateTime.UtcNow).ToString("O", CultureInfo.InvariantCulture)).Append(',');
            sb.Append(EscapeCsv(s.SpeakerKey)).Append(',');
            sb.Append(EscapeCsv(s.DisplayName ?? "")).Append(',');
            sb.Append(s.TrackId?.ToString(CultureInfo.InvariantCulture) ?? "").Append(',');
            sb.Append(s.AudioSpeakerId?.ToString(CultureInfo.InvariantCulture) ?? "");
            sb.AppendLine();
        }

        File.WriteAllText(path, sb.ToString(), Encoding.UTF8);
        return path;
    }

    private static string BuildHtml(MeetingSession session)
    {
        string title = $"Meeting Report — {session.StartedAtUtc:yyyy-MM-dd HH:mm:ss} UTC";

        (string htmlSpeaking, string htmlParticipation) = BuildLeaderboards(session);
        string htmlInterruptions = BuildInterruptions(session);
        string htmlInterruptionEvents = BuildInterruptionEvents(session);
        string htmlGraph = BuildConversationGraph(session);
        (string htmlEmotionOverall, string htmlEmotionBySpeaker) = BuildEmotions(session);
        string htmlTranscript = BuildTranscript(session);

        StringBuilder sb = new();
        sb.AppendLine("<!doctype html>");
        sb.AppendLine("<html>");
        sb.AppendLine("<head>");
        sb.AppendLine("  <meta charset=\"utf-8\" />");
        sb.AppendLine("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />");
        sb.Append("  <title>").Append(EscapeHtml(title)).AppendLine("</title>");
        sb.AppendLine("  <style>");
        sb.AppendLine("    :root {");
        sb.AppendLine("      --bg: #0b0f14;");
        sb.AppendLine("      --card: #121925;");
        sb.AppendLine("      --text: #e7eef7;");
        sb.AppendLine("      --muted: #9fb0c5;");
        sb.AppendLine("      --accent: #7dd3fc;");
        sb.AppendLine("      --border: rgba(255,255,255,0.08);");
        sb.AppendLine("    }");
        sb.AppendLine("    html, body { height: 100%; }");
        sb.AppendLine("    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--text); }");
        sb.AppendLine("    .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }");
        sb.AppendLine("    h1 { font-size: 22px; margin: 0 0 8px; }");
        sb.AppendLine("    .meta { color: var(--muted); font-size: 13px; margin-bottom: 18px; }");
        sb.AppendLine("    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }");
        sb.AppendLine("    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }");
        sb.AppendLine("    .card h2 { font-size: 15px; margin: 0 0 10px; color: var(--accent); }");
        sb.AppendLine("    table { width: 100%; border-collapse: collapse; font-size: 13px; }");
        sb.AppendLine("    th, td { padding: 8px 6px; border-bottom: 1px solid var(--border); text-align: left; }");
        sb.AppendLine("    th { color: var(--muted); font-weight: 600; }");
        sb.AppendLine("    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }");
        sb.AppendLine("    .transcript { white-space: pre-wrap; font-size: 13px; line-height: 1.4; max-height: 460px; overflow: auto; }");
        sb.AppendLine("    .small { color: var(--muted); font-size: 12px; }");
        sb.AppendLine("    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }");
        sb.AppendLine("  </style>");
        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("  <div class=\"wrap\">");
        sb.Append("    <h1>").Append(EscapeHtml(title)).AppendLine("</h1>");
        sb.Append("    <div class=\"meta\">Duration: <span class=\"mono\">")
            .Append(EscapeHtml(FormatDuration(session)))
            .Append("</span> · Segments: <span class=\"mono\">")
            .Append(session.Segments.Count.ToString(CultureInfo.InvariantCulture))
            .Append("</span> · Utterances: <span class=\"mono\">")
            .Append(session.Utterances.Count.ToString(CultureInfo.InvariantCulture))
            .AppendLine("</span></div>");

        sb.AppendLine("    <div class=\"grid\">");
        sb.AppendLine("      <div class=\"card\"><h2>Speaking Time</h2>");
        sb.AppendLine(htmlSpeaking);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Participation Score</h2><div class=\"small\">Heuristic: seconds + 5×turns − 3×interruptions</div>");
        sb.AppendLine(htmlParticipation);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Interruptions</h2>");
        sb.AppendLine(htmlInterruptions);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Interruption Events</h2><div class=\"small\">Overlap-based (who started while someone else was still speaking)</div>");
        sb.AppendLine(htmlInterruptionEvents);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Conversation Graph</h2>");
        sb.AppendLine(htmlGraph);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Emotion (Overall)</h2>");
        sb.AppendLine(htmlEmotionOverall);
        sb.AppendLine("      </div>");
        sb.AppendLine("      <div class=\"card\"><h2>Emotion (By Speaker)</h2>");
        sb.AppendLine(htmlEmotionBySpeaker);
        sb.AppendLine("      </div>");
        sb.AppendLine("    </div>");

        sb.AppendLine("    <div class=\"card\" style=\"margin-top: 14px;\"><h2>Transcript</h2>");
        sb.AppendLine(htmlTranscript);
        sb.AppendLine("    </div>");
        sb.AppendLine("  </div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");
        return sb.ToString();
    }

    private static (string Speaking, string Participation) BuildLeaderboards(MeetingSession session)
    {
        string speaking = BuildTable(
            headers: ["Speaker", "mm:ss"],
            rows: session.SpeakingTimeSecondsBySpeaker
                .OrderByDescending(kvp => kvp.Value)
                .Select(kvp => (Key: kvp.Key, Name: ResolveDisplay(session, kvp.Key), Value: TimeSpan.FromSeconds(kvp.Value).ToString("mm\\:ss", CultureInfo.InvariantCulture)))
                .ToList());

        string participation = BuildTable(
            headers: ["Speaker", "Score"],
            rows: session.ParticipationScoreBySpeaker
                .OrderByDescending(kvp => kvp.Value)
                .Select(kvp => (Key: kvp.Key, Name: ResolveDisplay(session, kvp.Key), Value: kvp.Value.ToString("0", CultureInfo.InvariantCulture)))
                .ToList());

        return (speaking, participation);
    }

    private static string BuildInterruptions(MeetingSession session)
    {
        if (session.InterruptionsBySpeaker.Count == 0)
        {
            return "<div class=\"small\">No interruptions detected.</div>";
        }

        return BuildTable(
            headers: ["Speaker", "Count"],
            rows: session.InterruptionsBySpeaker
                .OrderByDescending(kvp => kvp.Value)
                .Select(kvp => (Key: kvp.Key, Name: ResolveDisplay(session, kvp.Key), Value: kvp.Value.ToString(CultureInfo.InvariantCulture)))
                .ToList());
    }

    private static string BuildInterruptionEvents(MeetingSession session)
    {
        if (session.InterruptionEvents.Count == 0)
        {
            return "<div class=\"small\">No interruption overlaps detected.</div>";
        }

        StringBuilder sb = new();
        sb.Append("<table><thead><tr><th>Time</th><th>Interrupter</th><th>Interrupted</th><th>Overlap (s)</th></tr></thead><tbody>");
        foreach (InterruptionEvent e in session.InterruptionEvents
            .OrderBy(e => e.WhenUtc)
            .Take(30))
        {
            string time = e.WhenUtc.ToLocalTime().ToString("HH:mm:ss", CultureInfo.InvariantCulture);
            sb.Append("<tr><td class=\"mono\">").Append(EscapeHtml(time)).Append("</td>");
            sb.Append("<td>").Append(EscapeHtml(ResolveDisplay(session, e.InterrupterSpeakerKey))).Append("</td>");
            sb.Append("<td>").Append(EscapeHtml(ResolveDisplay(session, e.InterruptedSpeakerKey))).Append("</td>");
            sb.Append("<td class=\"mono\">").Append(e.OverlapSeconds.ToString("0.000", CultureInfo.InvariantCulture)).Append("</td></tr>");
        }
        sb.Append("</tbody></table>");
        return sb.ToString();
    }

    private static (string Overall, string BySpeaker) BuildEmotions(MeetingSession session)
    {
        string overall = BuildEmotionCountsTable(session.EmotionCountsOverall);

        if (session.EmotionCountsBySpeaker.Count == 0)
        {
            return (overall, "<div class=\"small\">No per-speaker emotion samples available.</div>");
        }

        List<(string Key, string Name, string Value)> rows = new();
        foreach ((string speakerKey, Dictionary<string, int> counts) in session.EmotionCountsBySpeaker)
        {
            if (counts.Count == 0)
            {
                continue;
            }

            (string emotion, int count) = counts.OrderByDescending(kvp => kvp.Value).First();
            rows.Add((speakerKey, ResolveDisplay(session, speakerKey), $"{emotion} ({count})"));
        }

        string bySpeaker = BuildTable(headers: ["Speaker", "Top Emotion"], rows: rows
            .OrderBy(r => r.Name, StringComparer.OrdinalIgnoreCase)
            .ToList());

        return (overall, bySpeaker);
    }

    private static string BuildEmotionCountsTable(Dictionary<string, int> counts)
    {
        if (counts.Count == 0)
        {
            return "<div class=\"small\">No emotion samples captured.</div>";
        }

        return BuildTable(
            headers: ["Emotion", "Count"],
            rows: counts
                .OrderByDescending(kvp => kvp.Value)
                .Select(kvp => (Key: kvp.Key, Name: kvp.Key, Value: kvp.Value.ToString(CultureInfo.InvariantCulture)))
                .ToList());
    }

    private static string BuildConversationGraph(MeetingSession session)
    {
        if (session.ConversationGraph.Count == 0)
        {
            return "<div class=\"small\">No edges computed.</div>";
        }

        StringBuilder sb = new();
        sb.Append("<table><thead><tr><th>From</th><th>To</th><th>Count</th></tr></thead><tbody>");
        foreach (ConversationEdge e in session.ConversationGraph.Take(25))
        {
            sb.Append("<tr><td>").Append(EscapeHtml(ResolveDisplay(session, e.FromSpeakerKey))).Append("</td>");
            sb.Append("<td>").Append(EscapeHtml(ResolveDisplay(session, e.ToSpeakerKey))).Append("</td>");
            sb.Append("<td class=\"mono\">").Append(e.Count.ToString(CultureInfo.InvariantCulture)).Append("</td></tr>");
        }
        sb.Append("</tbody></table>");
        return sb.ToString();
    }

    private static string BuildTranscript(MeetingSession session)
    {
        if (session.Utterances.Count == 0)
        {
            return "<div class=\"small\">No transcript available.</div>";
        }

        StringBuilder sb = new();
        sb.Append("<div class=\"transcript\">");
        foreach (Utterance u in session.Utterances.OrderBy(u => u.StartUtc))
        {
            string who = u.DisplayName ?? u.SpeakerKey;
            string time = u.StartUtc.ToLocalTime().ToString("HH:mm:ss", CultureInfo.InvariantCulture);
            sb.Append('[').Append(EscapeHtml(time)).Append("] ");
            sb.Append(EscapeHtml(who)).Append(": ");
            sb.Append(EscapeHtml(u.Text)).Append('\n');
        }
        sb.Append("</div>");
        return sb.ToString();
    }

    private static string BuildTable(string[] headers, List<(string Key, string Name, string Value)> rows)
    {
        StringBuilder sb = new();
        sb.Append("<table><thead><tr>");
        foreach (string h in headers)
        {
            sb.Append("<th>").Append(EscapeHtml(h)).Append("</th>");
        }
        sb.Append("</tr></thead><tbody>");

        foreach ((string Key, string Name, string Value) r in rows.Take(20))
        {
            sb.Append("<tr><td>").Append(EscapeHtml(string.IsNullOrWhiteSpace(r.Name) ? r.Key : r.Name)).Append("</td>");
            sb.Append("<td class=\"mono\">").Append(EscapeHtml(r.Value)).Append("</td></tr>");
        }

        sb.Append("</tbody></table>");
        return sb.ToString();
    }

    private static string ResolveDisplay(MeetingSession session, string speakerKey)
    {
        // Prefer a display name seen in segments/utterances.
        string? name =
            session.Segments.FirstOrDefault(s => string.Equals(s.SpeakerKey, speakerKey, StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(s.DisplayName))?.DisplayName
            ?? session.Utterances.FirstOrDefault(u => string.Equals(u.SpeakerKey, speakerKey, StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(u.DisplayName))?.DisplayName;

        return name ?? speakerKey;
    }

    private static string FormatDuration(MeetingSession session)
    {
        DateTime end = session.EndedAtUtc ?? DateTime.UtcNow;
        TimeSpan dur = end - session.StartedAtUtc;
        if (dur < TimeSpan.Zero) dur = TimeSpan.Zero;
        return dur.ToString("hh\\:mm\\:ss", CultureInfo.InvariantCulture);
    }

    private static string EscapeHtml(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return s.Replace("&", "&amp;", StringComparison.Ordinal)
            .Replace("<", "&lt;", StringComparison.Ordinal)
            .Replace(">", "&gt;", StringComparison.Ordinal)
            .Replace("\"", "&quot;", StringComparison.Ordinal);
    }

    private static string EscapeCsv(string s)
    {
        bool needsQuotes = false;
        for (int i = 0; i < s.Length; i++)
        {
            char c = s[i];
            if (c is '"' or ',' or '\n' or '\r')
            {
                needsQuotes = true;
                break;
            }
        }

        if (needsQuotes)
        {
            return "\"" + s.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
        }
        return s;
    }
}
