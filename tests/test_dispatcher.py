import asyncio
import pytest

from llmfleet import FleetDispatcher, RoutingPolicy


# ---------------- Fake Anthropic-shaped client ----------------

class FakeMessagesCreate:
    def __init__(self):
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"id": f"msg_{len(self.calls)}", "kwargs": kwargs}


class FakeBatchResults:
    def __init__(self, requests):
        self._requests = requests

    def __aiter__(self):
        async def gen():
            for r in self._requests:
                yield {"custom_id": r["custom_id"], "result": {"echo": r["params"]}}

        return gen()


class FakeBatches:
    def __init__(self):
        self.created = []
        self._results: dict[str, list] = {}
        self._counter = 0

    async def create(self, *, requests):
        self._counter += 1
        bid = f"batch_{self._counter}"
        self._results[bid] = list(requests)
        self.created.append((bid, requests))

        class B:
            pass

        b = B()
        b.id = bid
        return b

    async def retrieve(self, batch_id):
        # Always "ended" immediately for tests.
        return {"id": batch_id, "processing_status": "ended"}

    def results(self, batch_id):
        return FakeBatchResults(self._results[batch_id])


class FakeMessages:
    def __init__(self):
        self.create = FakeMessagesCreate()
        self.batches = FakeBatches()


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


# ---------------- Tests ----------------


@pytest.mark.asyncio
async def test_sync_routing_for_tight_latency():
    client = FakeAnthropic()
    policy = RoutingPolicy(sync_max_latency_ms=5000, poll_interval_s=0.01)
    async with FleetDispatcher(client, policy=policy) as fleet:
        result = await fleet.submit(
            latency_budget_ms=1000, model="m", messages=[{"role": "user", "content": "hi"}]
        )
    assert result["id"] == "msg_1"
    assert fleet.stats.sync_calls == 1
    assert fleet.stats.batched_calls == 0


@pytest.mark.asyncio
async def test_batch_routing_for_loose_latency():
    client = FakeAnthropic()
    policy = RoutingPolicy(
        sync_max_latency_ms=1000,
        batch_window_ms=100,  # short window for fast tests
        batch_min_size=1,
        poll_interval_s=0.01,
    )
    async with FleetDispatcher(client, policy=policy) as fleet:
        result = await fleet.submit(
            latency_budget_ms=600_000,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert result == {"echo": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}}
    assert fleet.stats.batched_calls == 1
    assert fleet.stats.batches_submitted == 1


@pytest.mark.asyncio
async def test_force_sync_overrides_policy():
    client = FakeAnthropic()
    async with FleetDispatcher(
        client, policy=RoutingPolicy(batch_window_ms=10_000, poll_interval_s=0.01)
    ) as fleet:
        await fleet.submit(force_sync=True, model="m", messages=[])
    assert fleet.stats.sync_calls == 1


@pytest.mark.asyncio
async def test_force_batch_overrides_policy():
    client = FakeAnthropic()
    async with FleetDispatcher(
        client, policy=RoutingPolicy(batch_window_ms=50, poll_interval_s=0.01)
    ) as fleet:
        await fleet.submit(force_batch=True, model="m", messages=[])
    assert fleet.stats.batched_calls == 1


@pytest.mark.asyncio
async def test_concurrent_submissions_pool_into_one_batch():
    client = FakeAnthropic()
    policy = RoutingPolicy(
        batch_window_ms=200,
        batch_min_size=3,
        batch_max_size=10,
        poll_interval_s=0.01,
    )
    async with FleetDispatcher(client, policy=policy) as fleet:
        results = await asyncio.gather(
            fleet.submit_batch(model="m", messages=[{"role": "user", "content": "a"}]),
            fleet.submit_batch(model="m", messages=[{"role": "user", "content": "b"}]),
            fleet.submit_batch(model="m", messages=[{"role": "user", "content": "c"}]),
        )
    assert len(results) == 3
    # All three should have been pooled into a single batch.
    assert fleet.stats.batches_submitted == 1
    assert fleet.stats.batched_calls == 3


@pytest.mark.asyncio
async def test_drain_on_close_flushes_pending():
    client = FakeAnthropic()
    policy = RoutingPolicy(batch_window_ms=10_000, batch_min_size=99, poll_interval_s=0.01)
    fleet = FleetDispatcher(client, policy=policy)
    await fleet.__aenter__()
    fut = asyncio.ensure_future(
        fleet.submit_batch(model="m", messages=[{"role": "user", "content": "x"}])
    )
    # Give the queue a moment to receive the item.
    await asyncio.sleep(0.05)
    await fleet.close()
    result = await fut
    assert result == {"echo": {"model": "m", "messages": [{"role": "user", "content": "x"}]}}


@pytest.mark.asyncio
async def test_on_batch_submitted_callback():
    client = FakeAnthropic()
    seen = []
    policy = RoutingPolicy(batch_window_ms=50, poll_interval_s=0.01)
    async with FleetDispatcher(
        client, policy=policy, on_batch_submitted=lambda b: seen.append(b)
    ) as fleet:
        await fleet.submit_batch(model="m", messages=[])
    assert len(seen) == 1
    assert seen[0].request_count == 1
