"""Phase 0 exit test: send one dummy trace to Langfuse and print its URL.

Run from the repo root:  python -m observability.hello_trace
"""
from observability.tracing import flush, new_trace_id, span
from langfuse import get_client

trace_id = new_trace_id()
with span("hello_request", trace_id=trace_id, input={"query": "hello"}):
    with span("dummy_stage", input="in") as s:
        s.update(output="out")

flush()
print("trace_id:", trace_id)
print("url:", get_client().get_trace_url(trace_id=trace_id))
