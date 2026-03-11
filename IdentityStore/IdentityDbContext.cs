using Microsoft.EntityFrameworkCore;

namespace IdentityStore
{
    /// <summary>
    /// The Entity Framework database context responsible for managing identity persistence.
    /// </summary>
    /// <remarks>
    /// Design Documentation
    /// 
    /// Purpose:
    /// Bridges the application's Person entity with the underlying SQLite database.
    ///
    /// Responsibilities:
    /// - Define the database schema for facial identities.
    /// - Configure the SQLite database source.
    /// - Provide access to the Persons dataset.
    ///
    /// Dependencies:
    /// - Microsoft.EntityFrameworkCore (ORM)
    /// - SQLite (Persistence engine)
    ///
    /// Architectural Role:
    /// Data Access Layer / DbContext.
    ///
    /// Constraints:
    /// - Hardcoded to "identity.db" for simplicity in this version.
    /// </remarks>
    public class IdentityDbContext : DbContext
    {
        /// <summary>
        /// Gets or sets the collection of registered persons in the database.
        /// </summary>
        public DbSet<Person> Persons => Set<Person>();

        /// <summary>
        /// Configures the database connection and options.
        /// </summary>
        /// <param name="options">The builder used to configure the context.</param>
        protected override void OnConfiguring(DbContextOptionsBuilder options)
        {
            options.UseSqlite("Data Source=identity.db");
        }
    }
}

