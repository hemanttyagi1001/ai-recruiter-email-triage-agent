"""
Runnable module — `python -m app.api` boots the FastAPI service.

Reads host/port from settings so binding is env-configurable.
"""

import uvicorn

from app.config import settings


def main() -> None:
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
