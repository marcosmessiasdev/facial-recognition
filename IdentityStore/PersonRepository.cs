using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;

namespace IdentityStore
{
    /// <summary>
    /// Repository class for managing Person entities and performing facial identity matching.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Encapsulates the logic for person persistence and visual identity lookup, 
    /// providing a clean API for the rest of the facial recognition system.
    ///
    /// Responsibilities:
    /// - Handle CRUD operations for the Person entity via Entity Framework.
    /// - Ensure the database schema is created on initialization.
    /// - Implementation of facial matching logic using Cosine Similarity between embeddings.
    /// - Managing the lifecycle of the underlying DbContext.
    ///
    /// Dependencies:
    /// - IdentityDbContext (Data Access)
    /// - Microsoft.EntityFrameworkCore (Database operations)
    ///
    /// Architectural Role:
    /// Repository / Data Access Object.
    ///
    /// Constraints:
    /// - Facial matching is currently performed in-memory (exhaustive search), which might 
    ///   scale poorly with thousands of registered individuals.
    /// </remarks>
    public class PersonRepository : IDisposable
    {
        private readonly IdentityDbContext _db;

        /// <summary>
        /// Initializes a new instance of the PersonRepository and ensures the database is ready.
        /// </summary>
        public PersonRepository()
        {
            _db = new IdentityDbContext();
            _db.Database.EnsureCreated(); // Creates the DB file and schema on first run
        }

        /// <summary>
        /// Registers a new person with their name and facial embedding.
        /// </summary>
        /// <param name="name">The full name of the person.</param>
        /// <param name="embedding">The 512-dimensional feature vector extracted from their face.</param>
        public void RegisterPerson(string name, float[] embedding)
        {
            var person = new Person { Name = name };
            person.Embedding = embedding;
            _db.Persons.Add(person);
            _db.SaveChanges();
        }

        /// <summary>
        /// Retrieves all registered persons from the database without tracking changes.
        /// </summary>
        /// <returns>A list of all Person entities.</returns>
        public List<Person> GetAll()
        {
            return _db.Persons.AsNoTracking().ToList();
        }

        /// <summary>
        /// Finds the best match for a given facial embedding using cosine similarity.
        /// </summary>
        /// <param name="queryEmbedding">
        /// The embedding of the face currently being analyzed.
        /// </param>
        /// <param name="threshold">
        /// The minimum similarity score (0 to 1) required to consider a match valid. 
        /// Default is 0.4.
        /// </param>
        /// <returns>
        /// A tuple containing the best matching Person (or null if none exceed threshold) 
        /// and the calculated similarity score.
        /// </returns>
        /// <remarks>
        /// This method iterates through the entire database and calculates the 
        /// cosine similarity for each registered person.
        /// </remarks>
        public (Person? person, float similarity) FindBestMatch(float[] queryEmbedding, float threshold = 0.4f)
        {
            var all = GetAll();
            Person? best = null;
            float bestSim = -1f;

            foreach (var p in all)
            {
                var emb = p.Embedding;
                if (emb == null || emb.Length != queryEmbedding.Length) continue;

                float sim = CosineSimilarity(queryEmbedding, emb);
                if (sim > bestSim)
                {
                    bestSim = sim;
                    best = p;
                }
            }

            return bestSim >= threshold ? (best, bestSim) : (null, bestSim);
        }

        /// <summary>
        /// Calculates the cosine similarity between two feature vectors.
        /// </summary>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>A similarity score where 1.0 means identical vectors and 0.0 means orthogonal.</returns>
        private static float CosineSimilarity(float[] a, float[] b)
        {
            float dot = 0f, magA = 0f, magB = 0f;
            for (int i = 0; i < a.Length; i++)
            {
                dot  += a[i] * b[i];
                magA += a[i] * a[i];
                magB += b[i] * b[i];
            }
            if (magA == 0 || magB == 0) return 0f;
            return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
        }

        /// <summary>
        /// Disposes the underlying database context.
        /// </summary>
        public void Dispose() => _db.Dispose();
    }
}

