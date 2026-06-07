import asyncio
import pytest

from llmfleet import DispatchStats, FleetDispatcher, RoutingPolicy


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
            latency_budget_ms=1000,
            model="m",
            messages=[{"role": "user", "content": "hi"}],
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
    assert result == {
        "echo": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    }
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
    policy = RoutingPolicy(
        batch_window_ms=10_000, batch_min_size=99, poll_interval_s=0.01
    )
    fleet = FleetDispatcher(client, policy=policy)
    await fleet.__aenter__()
    fut = asyncio.ensure_future(
        fleet.submit_batch(model="m", messages=[{"role": "user", "content": "x"}])
    )
    # Give the queue a moment to receive the item.
    await asyncio.sleep(0.05)
    await fleet.close()
    result = await fut
    assert result == {
        "echo": {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    }


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


# ---------------- Object-shaped client with configurable result payload ----------------


class _ObjResult:
    """Attribute-style batch result, like the real Anthropic SDK returns."""

    def __init__(self, custom_id, *, result=None, error=None):
        self.custom_id = custom_id
        self.result = result
        self.error = error


class _ObjBatches:
    def __init__(self, payload_for):
        # payload_for(custom_id) -> _ObjResult
        self._payload_for = payload_for
        self._requests = {}
        self._counter = 0

    async def create(self, *, requests):
        self._counter += 1
        bid = f"objbatch_{self._counter}"
        self._requests[bid] = list(requests)

        class B:
            pass

        b = B()
        b.id = bid
        return b

    async def retrieve(self, batch_id):
        # Attribute-style status object, again like the real SDK.
        class S:
            processing_status = "ended"

        return S()

    def results(self, batch_id):
        return [self._payload_for(r["custom_id"]) for r in self._requests[batch_id]]


def _make_obj_client(payload_for):
    class M:
        def __init__(self):
            self.create = FakeMessagesCreate()
            self.batches = _ObjBatches(payload_for)

    class C:
        def __init__(self):
            self.messages = M()

    return C()


@pytest.mark.asyncio
async def test_batch_result_with_falsy_payload_is_preserved():
    # Regression: an empty-dict (falsy but valid) payload must reach the caller
    # verbatim, not be swallowed and replaced by the raw wrapper object.
    client = _make_obj_client(lambda cid: _ObjResult(cid, result={}))
    policy = RoutingPolicy(batch_window_ms=30, poll_interval_s=0.001)
    async with FleetDispatcher(client, policy=policy) as fleet:
        result = await fleet.submit_batch(model="m", messages=[])
    assert result == {}


@pytest.mark.asyncio
async def test_batch_result_with_empty_string_payload_is_preserved():
    client = _make_obj_client(lambda cid: _ObjResult(cid, result=""))
    policy = RoutingPolicy(batch_window_ms=30, poll_interval_s=0.001)
    async with FleetDispatcher(client, policy=policy) as fleet:
        result = await fleet.submit_batch(model="m", messages=[])
    assert result == ""


@pytest.mark.asyncio
async def test_batch_result_error_propagates_as_exception():
    client = _make_obj_client(lambda cid: _ObjResult(cid, error="rate_limited"))
    policy = RoutingPolicy(batch_window_ms=30, poll_interval_s=0.001)
    async with FleetDispatcher(client, policy=policy) as fleet:
        with pytest.raises(RuntimeError, match="rate_limited"):
            await fleet.submit_batch(model="m", messages=[])


@pytest.mark.asyncio
async def test_submit_after_close_raises():
    client = FakeAnthropic()
    fleet = FleetDispatcher(client, policy=RoutingPolicy(poll_interval_s=0.01))
    await fleet.__aenter__()
    await fleet.close()
    with pytest.raises(RuntimeError, match="closed"):
        await fleet.submit_sync(model="m", messages=[])


def test_dispatch_stats_total():
    stats = DispatchStats(sync_calls=2, batched_calls=5)
    assert stats.total == 7


def test_routing_policy_cost_aware_applies_overrides():
    policy = RoutingPolicy.cost_aware(batch_window_ms=1234, batch_max_size=7)
    assert policy.batch_window_ms == 1234
    assert policy.batch_max_size == 7
    # Untouched fields keep their defaults.
    assert policy.sync_max_latency_ms == 5_000
