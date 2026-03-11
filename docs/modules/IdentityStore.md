# Module: IdentityStore

## Purpose
Manages the registration and lifecycle of known individuals.

## Key Components
- **Person**: The core domain entity for a registered individual.
- **PersonRepository**: Data access layer for CRUD and search.
- **IdentityDbContext**: Entity Framework configuration for SQLite.

## Responsibilities
- Persist names and facial embeddings to a local SQLite database (`identity.db`).
- Perform high-performance cosine similarity searches across all registered embeddings.
- Handle database schema initialization (`EnsureCreated`).

## Dependencies
- **Microsoft.EntityFrameworkCore.Sqlite**: Storage engine.
- **System.Text.Json**: Used internally for serializing vector data.
