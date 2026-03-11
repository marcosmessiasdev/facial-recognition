namespace SpeakerDiarization;

/// <summary>
/// State machine for real-time audio speaker deduplication and clustering.
/// </summary>
/// <remarks>
/// Design Documentation
/// 
/// Purpose:
/// Processes chunks of raw audio, extracting voice embeddings, and clustering them 
/// on the fly. Assigns temporary unique integer IDs to distinct voices.
///
/// Responsibilities:
/// - Buffer active speech segments based on Voice Activity Detection (VAD) signals.
/// - Break speech into sliding windows for embedding extraction via SpeakerEmbeddingExtractor.
/// - Iteratively build an in-memory cosine-similarity cluster of known speaker vectors.
///
/// Dependencies:
/// - SpeakerEmbeddingExtractor (ONNX TitaNet/SpeakerNet models).
///
/// Architectural Role:
/// Audio Domain Service / Streaming Diarization Engine.
/// </remarks>
public sealed class OnlineSpeakerDiarizer(string embeddingModelPath, int sampleRateHz = 16000, int windowMs = 1500, int hopMs = 750, float assignThreshold = 0.62f, int hangoverMs = 350) : IDisposable
{
    private readonly SpeakerEmbeddingExtractor _embedder = new(embeddingModelPath, sampleRateHz);
    private readonly int _sampleRateHz = sampleRateHz;
    private readonly int _windowSamples = (int)(sampleRateHz * (windowMs / 1000.0));
    private readonly int _hopSamples = (int)(sampleRateHz * (hopMs / 1000.0));
    private readonly float _assignThreshold = Math.Clamp(assignThreshold, 0.3f, 0.95f);

    private readonly List<SpeakerCluster> _clusters = new();
    private readonly List<float> _speechBuffer = new();
    private TimeSpan _speechStart;
    private bool _inSpeech;
    private TimeSpan _lastSpeechTime;
    private readonly TimeSpan _hangover = TimeSpan.FromMilliseconds(Math.Clamp(hangoverMs, 0, 2000));

    /// <summary>
    /// Triggered when a grouped chunk of speech is finalized and assigned a specific cluster ID.
    /// </summary>
    public event EventHandler<AudioSpeakerSegment>? SegmentReady;

    /// <summary>
    /// Processes incoming raw audio samples under the guidance of an external VAD.
    /// </summary>
    /// <param name="samples16kMono">Audio chunks (usually 512-1024 samples) at 16kHz Mono.</param>
    /// <param name="timestamp">Chronological start stamp of the buffer.</param>
    /// <param name="speechActive">True if speech was detected concurrently.</param>
    public void PushFrame(float[] samples16kMono, TimeSpan timestamp, bool speechActive)
    {
        if (speechActive)
        {
            if (!_inSpeech)
            {
                _inSpeech = true;
                _speechStart = timestamp;
                _speechBuffer.Clear();
            }
            _lastSpeechTime = timestamp;
        }

        if (_inSpeech)
        {
            _speechBuffer.AddRange(samples16kMono);

            TimeSpan silentFor = timestamp - _lastSpeechTime;
            if (!speechActive && silentFor >= _hangover)
            {
                FinalizeSpeech(timestamp);
                _inSpeech = false;
            }
        }
    }

    /// <summary>
    /// Closes the current contiguous speech buffer, chunks it for embeddings, and emits segments.
    /// </summary>
    /// <param name="end">Timeline mark when speech finalized.</param>
    private void FinalizeSpeech(TimeSpan end)
    {
        float[] pcm = _speechBuffer.ToArray();
        if (pcm.Length < _windowSamples)
        {
            return;
        }

        // Sliding windows with overlap.
        List<(TimeSpan Start, TimeSpan End, int SpeakerId, float Score)> assignments = new();
        for (int offset = 0; offset + _windowSamples <= pcm.Length; offset += _hopSamples)
        {
            float[] window = new float[_windowSamples];
            Array.Copy(pcm, offset, window, 0, _windowSamples);

            float[] emb = _embedder.GetEmbedding(window);
            if (emb.Length == 0)
            {
                continue;
            }

            (int sid, float score) = Assign(emb);

            TimeSpan start = _speechStart + TimeSpan.FromSeconds(offset / (double)_sampleRateHz);
            TimeSpan segEnd = start + TimeSpan.FromSeconds(_windowSamples / (double)_sampleRateHz);
            assignments.Add((start, segEnd, sid, score));
        }

        // Merge adjacent windows of same speaker.
        foreach (AudioSpeakerSegment seg in Merge(assignments))
        {
            SegmentReady?.Invoke(this, seg);
        }
    }

    private (int SpeakerId, float Score) Assign(float[] emb)
    {
        if (_clusters.Count == 0)
        {
            SpeakerCluster c = new(id: 1, emb);
            _clusters.Add(c);
            return (c.Id, 1f);
        }

        int bestId = -1;
        float best = -1f;
        foreach (SpeakerCluster c in _clusters)
        {
            float s = Cosine(emb, c.Centroid);
            if (s > best)
            {
                best = s;
                bestId = c.Id;
            }
        }

        if (best >= _assignThreshold && bestId > 0)
        {
            _clusters.First(c => c.Id == bestId).Update(emb);
            return (bestId, best);
        }

        int newId = _clusters.Max(c => c.Id) + 1;
        SpeakerCluster nc = new(newId, emb);
        _clusters.Add(nc);
        return (newId, best);
    }

    private static IEnumerable<AudioSpeakerSegment> Merge(List<(TimeSpan Start, TimeSpan End, int SpeakerId, float Score)> segs)
    {
        if (segs.Count == 0)
        {
            yield break;
        }

        segs.Sort((a, b) => a.Start.CompareTo(b.Start));

        (TimeSpan Start, TimeSpan End, int SpeakerId, float Score) cur = segs[0];
        for (int i = 1; i < segs.Count; i++)
        {
            (TimeSpan Start, TimeSpan End, int SpeakerId, float Score) s = segs[i];
            if (s.SpeakerId == cur.SpeakerId && s.Start <= cur.End + TimeSpan.FromMilliseconds(50))
            {
                cur = (cur.Start, s.End, cur.SpeakerId, Math.Max(cur.Score, s.Score));
            }
            else
            {
                yield return new AudioSpeakerSegment(cur.Start, cur.End, cur.SpeakerId, cur.Score);
                cur = s;
            }
        }
        yield return new AudioSpeakerSegment(cur.Start, cur.End, cur.SpeakerId, cur.Score);
    }

    private static float Cosine(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return na <= 1e-12 || nb <= 1e-12 ? 0f : (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb)));
    }

    /// <summary>
    /// Releases the ONNX provider instance inside the underlying embedder.
    /// </summary>
    public void Dispose()
    {
        _embedder.Dispose();
    }

    private sealed class SpeakerCluster(int id, float[] emb)
    {
        public int Id { get; } = id;
        public float[] Centroid { get; private set; } = (float[])emb.Clone();
        private int _count = 1;

        public void Update(float[] emb)
        {
            _count++;
            float alpha = 1f / _count;
            for (int i = 0; i < Centroid.Length && i < emb.Length; i++)
            {
                Centroid[i] = ((1 - alpha) * Centroid[i]) + (alpha * emb[i]);
            }
        }
    }
}

