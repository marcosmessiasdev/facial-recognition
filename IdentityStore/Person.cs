using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace IdentityStore
{
    /// <summary>
    /// Represents a registered individual in the facial recognition system.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Persistent entity used to store a person's name and their corresponding facial feature vector (embedding).
    ///
    /// Responsibilities:
    /// - Store the database identifier and name of the person.
    /// - Handle the serialization of 512-dimensional float vectors into a database-friendly string format.
    /// - Provide a non-mapped property for working with raw float arrays in the application.
    ///
    /// Dependencies:
    /// - System.ComponentModel.DataAnnotations (Persistence metadata)
    ///
    /// Architectural Role:
    /// Domain Model / Entity.
    ///
    /// Constraints:
    /// - The embedding must typically be a 512-dimensional vector to match the ArcFace signature.
    /// </remarks>
    public class Person
    {
        /// <summary>
        /// Gets or sets the primary key for the person record.
        /// </summary>
        [Key]
        [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int Id { get; set; }

        /// <summary>
        /// Gets or sets the name of the recognized individual.
        /// </summary>
        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = "";

        /// <summary>
        /// Gets or sets the raw serialized string of facial feature embeddings.
        /// </summary>
        /// <remarks>
        /// Stored in the database as a comma-separated list of 512 floats to avoid 
        /// the complexity of BLOB storage or custom SQLite vector extensions.
        /// </remarks>
        public string EmbeddingJson { get; set; } = "";

        /// <summary>
        /// Gets or sets the facial feature embedding as a float array.
        /// This property is not mapped to the database.
        /// </summary>
        [NotMapped]
        public float[]? Embedding
        {
            get
            {
                if (string.IsNullOrEmpty(EmbeddingJson)) return null;
                var parts = EmbeddingJson.Split(',');
                var result = new float[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                    float.TryParse(parts[i], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out result[i]);
                return result;
            }
            set
            {
                if (value == null) { EmbeddingJson = ""; return; }
                EmbeddingJson = string.Join(",", System.Array.ConvertAll(value, 
                    v => v.ToString("G9", System.Globalization.CultureInfo.InvariantCulture)));
            }
        }
    }
}

