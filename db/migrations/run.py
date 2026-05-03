"""
Migration runner — applies db/schema.sql to the target database.

Usage:
    python -m db.migrations.run

Behaviour:
    - Idempotent: schema.sql uses IF NOT EXISTS throughout, safe to re-run.
    - Uses psycopg (sync) — no async complexity needed for a CLI script.
    - Reads DATABASE_URL from .env (falls back to the default local URL).
    - Prints a clear summary of what was applied.

When to run:
    - Fresh environment setup.
    - After pulling changes that modify schema.sql.
    - To verify schema is in sync with the codebase.
"""

import sys
from pathlib import Path

# Make sure project root is on the path when run as a script
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import psycopg
from dotenv import load_dotenv

from core.config import get_settings

SCHEMA_PATH = project_root / "db" / "schema.sql"


def _sync_url(async_url: str) -> str:
    """Convert asyncpg URL to psycopg-compatible URL."""
    return (
        async_url
        .replace("postgresql+asyncpg://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
    )


def run() -> None:
    load_dotenv(project_root / ".env")
    settings = get_settings()

    if not SCHEMA_PATH.exists():
        print(f"[ERROR] schema.sql not found at {SCHEMA_PATH}")
        sys.exit(1)

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    conn_url   = _sync_url(settings.database_url)

    print(f"Connecting to: {conn_url.split('@')[-1]}")   # print host/db only, hide credentials
    print(f"Applying:      {SCHEMA_PATH}")
    print("-" * 60)

    try:
        with psycopg.connect(conn_url) as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
    except psycopg.OperationalError as e:
        print(f"[ERROR] Could not connect to database: {e}")
        print("        Is the Postgres container running?  docker-compose up -d postgres")
        sys.exit(1)
    except psycopg.Error as e:
        print(f"[ERROR] Schema execution failed: {e}")
        sys.exit(1)

    # Verify tables were created
    try:
        with psycopg.connect(conn_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename;
                """)
                tables = [row[0] for row in cur.fetchall()]
    except psycopg.Error:
        tables = []

    expected = {
        "documents", "sections", "chunks", "assets",
        "relationships", "retrieval_logs", "evaluation_logs",
    }

    print("Tables:")
    all_ok = True
    for table in sorted(expected):
        status = "OK" if table in tables else "MISSING"
        mark   = "✓" if table in tables else "✗"
        print(f"  {mark}  {table:<25} {status}")
        if table not in tables:
            all_ok = False

    print("-" * 60)
    if all_ok:
        print("Migration complete. All tables present.")
    else:
        print("[WARN] Some tables are missing — check schema.sql for errors.")
        sys.exit(1)


if __name__ == "__main__":
    run()
