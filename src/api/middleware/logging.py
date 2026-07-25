"""
Request Logging Middleware.
Injects unique Request IDs and logs timing/status for all incoming HTTP requests.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Awaitable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique X-Request-ID header to every response
    and logs the request duration, method, path, and status code.
    """
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid.uuid4())
        
        # Inject request ID into the request state for use in routers if needed
        request.state.request_id = request_id

        start_time = time.perf_counter()
        
        try:
            response = await call_next(request)
            process_time = time.perf_counter() - start_time
            
            # Log success
            logger.info(
                f"{request.method} {request.url.path} "
                f"[{response.status_code}] "
                f"{process_time * 1000:.2f}ms "
                f"(ID: {request_id})"
            )
            
            # Add to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as exc:
            process_time = time.perf_counter() - start_time
            # Log unhandled exceptions
            logger.error(
                f"{request.method} {request.url.path} "
                f"[500] "
                f"{process_time * 1000:.2f}ms "
                f"(ID: {request_id}) - {exc!r}"
            )
            raise
