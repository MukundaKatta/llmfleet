# llmfleet

Fleet-level batch dispatcher for LLM APIs. Pool requests from many coroutines, route to provider Batch APIs (50% discount on Anthropic, OpenAI), keep your sync agent loops working.

```bash
pip install llmfleet
```

## Why

Anthropic's Batch API saves 50% on input tokens, but it's [terrible for one agent](https://eran.sandler.co.il/post/2026-04-27-batch-api-is-terrible-for-one-agent/) — single requests poll for 90–120s. The right unit of batching isn't one user's turn; it's a fleet of agents' turns pooled together by a layer the user never sees. `llmfleet` is that layer.

## Quick start

```python
import asyncio
from anthropic import AsyncAnthropic
from llmfleet import FleetDispatcher, RoutingPolicy

async def main():
    client = AsyncAnthropic()
    policy = RoutingPolicy(
        sync_max_latency_ms=5_000,    # interactive paths stay sync
        batch_window_ms=30_000,       # otherwise pool for 30s
        batch_min_size=10,
        batch_max_size=100,
    )

    async with FleetDispatcher(client, policy=policy) as fleet:
        # Tight latency → sync
        chat = await fleet.submit(
            latency_budget_ms=2_000,
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Loose latency → pooled into a batch
        graded = await asyncio.gather(*[
            fleet.submit(
                latency_budget_ms=600_000,
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": f"Grade: {essay}"}],
            )
            for essay in essays
        ])

asyncio.run(main())
```

## How it works

`FleetDispatcher` runs a background flusher coroutine. Calls to `submit()` either:

- **Run synchronously** if `latency_budget_ms <= policy.sync_max_latency_ms`, or
- **Get queued.** When the queue holds `batch_min_size` items or `batch_window_ms` elapses, the flusher submits one Anthropic Message Batch, polls until completion, and dispatches results back to each awaiting coroutine via its Future.

Concurrent `submit()` calls from independent coroutines automatically share batches.

## API

```python
FleetDispatcher(client, policy=None, on_batch_submitted=None)

# Lifecycle: use as async context manager
async with FleetDispatcher(client) as fleet:
    response = await fleet.submit(latency_budget_ms=N, **anthropic_messages_create_kwargs)

# Force routing decisions
await fleet.submit_sync(**kwargs)
await fleet.submit_batch(**kwargs)

# Introspection
fleet.stats.sync_calls
fleet.stats.batched_calls
fleet.stats.batches_submitted
```

## Configuration

```python
RoutingPolicy(
    sync_max_latency_ms=5_000,    # threshold for sync routing
    batch_window_ms=30_000,       # how long to wait for a batch to fill
    batch_min_size=1,             # minimum size before flushing early
    batch_max_size=100,           # hard cap (Anthropic supports up to 10k)
    poll_interval_s=2.0,          # batch status poll interval
)
```

## What it doesn't do

- Not a router across providers/models for quality. Use a real router.
- Not cross-process pooling — fleet is process-local. Use a shared queue (Redis / SQS) for cross-process.
- Doesn't try to batch tool-call turns where the tool is on the critical path; pass `force_sync=True` for those.

## Status

v0.1.0: Anthropic only. OpenAI Batch API and Bedrock async-invoke are on the roadmap. Patches welcome.

## License

MIT
