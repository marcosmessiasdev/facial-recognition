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
    private int _samplesProcessedInSpeech;
    private int _lastWindowStartSample;
    private int _currentSpeakerId;
    private float _currentSpeakerScore;
    private readonly Queue<(int SpeakerId, float Score)> _recentAssignments = new();
    private const int RecentAssignmentWindow = 5;

    // Guards against speaker ID fragmentation when embeddings are noisy.
    private int _pendingNewWindows;
    private int _pendingNewBestId;
    private float _pendingNewBestScore;
    private const int NewClusterConfirmWindows = 3;
    private const float NearThresholdMargin = 0.06f;
    private const float NearThresholdUpdateWeight = 0.05f;
    private const float MinRmsForEmbedding = 0.003f;

    /// <summary>
    /// Triggered when a grouped chunk of speech is finalized and assigned a specific cluster ID.
    /// </summary>
    public event EventHandler<AudioSpeakerSegment>? SegmentReady;

    /// <summary>
    /// Fired during active speech when a best-effort current speaker ID is updated (low latency).
    /// </summary>
    public event EventHandler<(TimeSpan Offset, int SpeakerId, float Confidence)>? ActiveSpeakerUpdated;

    /// <summary>
    /// Processes incoming raw audio samples under the guidance of an external VAD.
    /// </summary>
    /// <param name="samples16kMono">Audio chunks (usually 512-1024 samples) at 16kHz Mono.</param>
    /// <param name="timestamp">Chronological start stamp of the buffer.</param>
    /// <param name="speechActive">True if speech was detected concurrently.</param>
    public void PushFrame(float[] samples16kMono, TimeSpan timestamp, bool speechActive)
    {
        ArgumentNullException.ThrowIfNull(samples16kMono);

        if (speechActive)
        {
            if (!_inSpeech)
            {
                _inSpeech = true;
                _speechStart = timestamp;
                _speechBuffer.Clear();
                _samplesProcessedInSpeech = 0;
                _lastWindowStartSample = 0;
                _currentSpeakerId = 0;
                _currentSpeakerScore = 0;
                _recentAssignments.Clear();
                _pendingNewWindows = 0;
                _pendingNewBestId = 0;
                _pendingNewBestScore = 0f;
            }
            _lastSpeechTime = timestamp;
        }

        if (_inSpeech)
        {
            _speechBuffer.AddRange(samples16kMono);
            _samplesProcessedInSpeech += samples16kMono.Length;

            // Low-latency incremental assignment: as soon as we have a full window, compute/assign every hop.
            while (_samplesProcessedInSpeech - _lastWindowStartSample >= _windowSamples)
            {
                float[] window = new float[_windowSamples];
                int srcOffset = Math.Max(0, _samplesProcessedInSpeech - _windowSamples);
                // Use the most recent window for responsiveness.
                for (int i = 0; i < _windowSamples && (srcOffset + i) < _speechBuffer.Count; i++)
                {
                    window[i] = _speechBuffer[srcOffset + i];
                }

                if (Rms(window) >= MinRmsForEmbedding)
                {
                    float[] emb = _embedder.GetEmbedding(window);
                    if (emb.Length > 0)
                    {
                        (int sid, float score) = Assign(emb);
                        PushRecentAssignment(sid, score);
                        (int stableId, float stableScore) = GetStableRecentAssignment();
                        if (stableId != _currentSpeakerId || MathF.Abs(stableScore - _currentSpeakerScore) > 0.05f)
                        {
                            _currentSpeakerId = stableId;
                            _currentSpeakerScore = stableScore;
                            ActiveSpeakerUpdated?.Invoke(this, (timestamp, stableId, stableScore));
                        }
                    }
                }

                _lastWindowStartSample += _hopSamples;
                if (_lastWindowStartSample < 0)
                {
                    _lastWindowStartSample = 0;
                }

                // Avoid unbounded buffer growth: keep only last few seconds.
                int maxKeep = Math.Max(_windowSamples * 4, _windowSamples + _hopSamples);
                if (_speechBuffer.Count > maxKeep)
                {
                    int remove = _speechBuffer.Count - maxKeep;
                    _speechBuffer.RemoveRange(0, remove);
                    _samplesProcessedInSpeech -= remove;
                    _lastWindowStartSample = Math.Max(0, _lastWindowStartSample - remove);
                }
            }

            TimeSpan silentFor = timestamp - _lastSpeechTime;
            if (!speechActive && silentFor >= _hangover)
            {
                FinalizeSpeech(timestamp);
                _inSpeech = false;
            }
        }
    }

    /// <summary>
    /// Finalizes any in-progress speech buffer (best-effort) when the stream ends.
    /// </summary>
    public void Flush(TimeSpan endTimestamp)
    {
        if (_inSpeech)
        {
            FinalizeSpeech(endTimestamp);
            _inSpeech = false;
            _speechBuffer.Clear();
            _samplesProcessedInSpeech = 0;
            _lastWindowStartSample = 0;
            _currentSpeakerId = 0;
            _currentSpeakerScore = 0;
            _recentAssignments.Clear();
            _pendingNewWindows = 0;
            _pendingNewBestId = 0;
            _pendingNewBestScore = 0f;
        }
    }

    /// <summary>
    /// Closes the current contiguous speech buffer, chunks it for embeddings, and emits segments.
    /// </summary>
    /// <param name="end">Timeline mark when speech finalized.</param>
    private void FinalizeSpeech(TimeSpan end)
    {
        float[] pcm = [.. _speechBuffer];
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

            if (Rms(window) < MinRmsForEmbedding)
            {
                continue;
            }

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

        MergeSimilarClusters();
    }

    private (int SpeakerId, float Score) Assign(float[] emb)
    {
        if (_clusters.Count == 0)
        {
            SpeakerCluster c = new(id: 1, emb);
            _clusters.Add(c);
            _pendingNewWindows = 0;
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

        if (bestId > 0 && best >= _assignThreshold)
        {
            _pendingNewWindows = 0;
            _clusters.First(c => c.Id == bestId).UpdateWeighted(emb, weight: 1f);
            return (bestId, best);
        }

        if (bestId > 0 && best >= (_assignThreshold - NearThresholdMargin))
        {
            _pendingNewWindows = 0;
            _clusters.First(c => c.Id == bestId).UpdateWeighted(emb, weight: NearThresholdUpdateWeight);
            return (bestId, best);
        }

        _pendingNewWindows++;
        _pendingNewBestId = bestId;
        _pendingNewBestScore = best;

        if (_pendingNewWindows >= NewClusterConfirmWindows)
        {
            _pendingNewWindows = 0;
            int newId = _clusters.Max(c => c.Id) + 1;
            SpeakerCluster nc = new(newId, emb);
            _clusters.Add(nc);
            return (newId, Math.Max(0f, best));
        }

        return (bestId > 0 ? bestId : 1, Math.Max(0f, best));
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

    private void PushRecentAssignment(int speakerId, float score)
    {
        _recentAssignments.Enqueue((speakerId, score));
        while (_recentAssignments.Count > RecentAssignmentWindow)
        {
            _ = _recentAssignments.Dequeue();
        }
    }

    private (int SpeakerId, float Score) GetStableRecentAssignment()
    {
        if (_recentAssignments.Count == 0)
        {
            return (0, 0f);
        }

        Dictionary<int, (int Count, float BestScore)> counts = new();
        foreach ((int SpeakerId, float Score) a in _recentAssignments)
        {
            if (!counts.TryGetValue(a.SpeakerId, out (int Count, float BestScore) cur))
            {
                counts[a.SpeakerId] = (1, a.Score);
            }
            else
            {
                counts[a.SpeakerId] = (cur.Count + 1, Math.Max(cur.BestScore, a.Score));
            }
        }

        KeyValuePair<int, (int Count, float BestScore)> best = counts
            .OrderByDescending(kvp => kvp.Value.Count)
            .ThenByDescending(kvp => kvp.Value.BestScore)
            .First();
        return (best.Key, best.Value.BestScore);
    }

    private void MergeSimilarClusters()
    {
        if (_clusters.Count < 2)
        {
            return;
        }

        float mergeThreshold = Math.Min(0.9f, _assignThreshold + 0.16f);

        bool mergedAny;
        do
        {
            mergedAny = false;
            for (int i = 0; i < _clusters.Count; i++)
            {
                for (int j = i + 1; j < _clusters.Count; j++)
                {
                    SpeakerCluster a = _clusters[i];
                    SpeakerCluster b = _clusters[j];
                    float s = Cosine(a.Centroid, b.Centroid);
                    if (s < mergeThreshold)
                    {
                        continue;
                    }

                    SpeakerCluster keep = a.Count >= b.Count ? a : b;
                    SpeakerCluster drop = a.Count >= b.Count ? b : a;
                    keep.UpdateWeighted(drop.Centroid, weight: 0.25f);
                    _clusters.Remove(drop);
                    mergedAny = true;
                    break;
                }

                if (mergedAny)
                {
                    break;
                }
            }
        } while (mergedAny && _clusters.Count >= 2);
    }

    private static float Rms(float[] x)
    {
        if (x.Length == 0) return 0f;
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double v = x[i];
            sum += v * v;
        }
        return (float)Math.Sqrt(sum / x.Length);
    }

    private sealed class SpeakerCluster(int id, float[] emb)
    {
        public int Id { get; } = id;
        public float[] Centroid { get; private set; } = (float[])emb.Clone();
        private int _count = 1;

        public int Count => _count;

        public void UpdateWeighted(float[] emb, float weight)
        {
            weight = Math.Clamp(weight, 0f, 1f);
            if (weight <= 0f)
            {
                return;
            }

            _count++;
            for (int i = 0; i < Centroid.Length && i < emb.Length; i++)
            {
                Centroid[i] = ((1 - weight) * Centroid[i]) + (weight * emb[i]);
            }
        }
    }
}
