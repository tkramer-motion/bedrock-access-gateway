import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Generator

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# The shared ALB drops a connection that sends nothing for 60s, which happens while tools run or the model thinks
KEEPALIVE_INTERVAL = 15
KEEPALIVE = b": keep-alive\n\n"

# Tool loops can hold a thread for as long as a Lambda timeout (up to 900s), so streams get their own pool
executor = ThreadPoolExecutor(max_workers=64, thread_name_prefix="chat-stream")

_end = object()


def error_event(e: Exception) -> bytes:
    message = e.detail if isinstance(e, HTTPException) else str(e) or e.__class__.__name__
    return "data: {}\n\n".format(json.dumps({"error": {"message": message, "type": e.__class__.__name__}})).encode("utf-8")


async def keepalive_stream(chunks: Generator[bytes, None, None]) -> AsyncIterator[bytes]:
    """Stream a blocking generator without stalling the event loop.

    Sends an SSE comment while the generator is busy, and if the generator raises, ends the stream with an
    error event instead of dropping the connection (which clients report as an incomplete chunked payload).
    """
    loop = asyncio.get_running_loop()
    pending = None
    try:
        while True:
            pending = loop.run_in_executor(executor, next, chunks, _end)
            while not (await asyncio.wait({pending}, timeout=KEEPALIVE_INTERVAL))[0]:
                yield KEEPALIVE
            done, pending = pending, None
            chunk = done.result()
            if chunk is _end:
                return
            yield chunk
    except Exception as e:
        logger.exception("Chat stream failed")
        yield error_event(e)
        yield b"data: [DONE]\n\n"
    finally:
        if pending is not None:
            # The client went away while the generator was producing an item; close it once that is done
            pending.add_done_callback(lambda future: _close_abandoned(future, chunks))
        else:
            chunks.close()


def _close_abandoned(future: asyncio.Future, chunks: Generator[bytes, None, None]):
    if not future.cancelled() and future.exception() is not None:
        logger.info(f"Chat stream failed after the client disconnected: {future.exception()!r}")
    chunks.close()
