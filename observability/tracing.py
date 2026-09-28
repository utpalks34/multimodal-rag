"""trace_id creation and a span helper for each pipeline stage. (Phase 0)

Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from the environment.
"""
from contextlib import contextmanager

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()


def new_trace_id() -> str:
    """Create the trace_id carried through every stage of one request."""
    return get_client().create_trace_id()


@contextmanager
def span(name: str, trace_id: str | None = None, **kwargs):
    """Open a span. Pass trace_id on the top-level span of a request; nested spans inherit it."""
    ctx = {"trace_id": trace_id} if trace_id else None
    with get_client().start_as_current_observation(name=name, trace_context=ctx, **kwargs) as s:
        yield s


def flush() -> None:
    get_client().flush()
