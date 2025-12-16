#!/usr/bin/env python3
"""Initialize the database schema."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.database import Database
from src.utils.config import get_settings
from src.utils.logging import setup_logging, get_logger


async def main():
    """Initialize the database."""
    setup_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    db_path = settings.database_path

    logger.info(f"Initializing database at {db_path}")

    # Ensure data directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    await db.initialize()

    logger.info("Database initialized successfully")

    # Verify tables were created
    result = await db._fetchall(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    )
    tables = [r[0] for r in result]
    logger.info(f"Created tables: {', '.join(tables)}")

    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
