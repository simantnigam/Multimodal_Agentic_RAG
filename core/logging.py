import sys
from loguru import logger
from loguru import Logger

from core.config import get_settings


def setup_logging() -> None:
    """Configure loguru for structured JSON output. Call once at app startup."""
    settings = get_settings()

    logger.remove()

    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        serialize=True,     # emits newline-delimited JSON; format param not used with serialize
        backtrace=False,
        diagnose=False,     # never expose local variable values in prod
    )


def get_logger(name: str) -> Logger:
    """Return a loguru logger bound with the caller's module name."""
    return logger.bind(module=name)
