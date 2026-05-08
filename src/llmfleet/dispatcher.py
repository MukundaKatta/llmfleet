"""Fleet-level batch dispatcher.

Pools `messages.create`-style calls from many coroutines, submits them as a
single Anthropic Message Batch when the batch window fills or expires, polls
for completion, and dispatches results back to the awaiting coroutines.

Inspired by Eran Sandler's observation that the right unit of batching is a
fleet of agents, not a single user's turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional


@dataclass
class RoutingPolicy:
    """Policy controlling sync vs batch routing decisions.

    - `sync_max_latency_ms`: if a request's latency budget is tighter than
      this, route synchronously regardless of batch eligibility.
    - `batch_window_ms`: how long the dispatcher waits to fill a batch before
      submitting whatever it has.
    - `batch_min_size`: the smallest batch it will submit (smaller batches
      flush at the end of the window).
    - `batch_max_size`: hard cap; submit immediately when reached.
    - `poll_interval_s`: seconds between polls of batch status.
    """

    sync_max_latency_ms: int = 5_000
    batch_window_ms: int = 30_000
    batch_min_size: int = 1
    batch_max_size: int = 100
    poll_interval_s: float = 2.0

    @classmethod
    def cost_aware(cls, **overrides: Any) -> RoutingPolicy:
        return cls(**overrides)


@dataclass
class DispatchStats:
    sync_calls: int = 0
    batched_calls: int = 0
    batches_submitted: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.sync_calls + self.batched_calls


@dataclass
class _PendingItem:
    request_id: str
    params: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)


_SHUTDOWN = object()  # sentinel placed on the queue to signal clean shutdown


@dataclass
class PendingBatch:
    """A batch in-flight: useful for introspection."""

    batch_id: str
    request_count: int
    submitted_at: float


@dataclass
class BatchResult:
    """A single batch result returned to the caller via `submit`."""

    request_id: str
    response: Any
    error: Optional[Exception] = None


class FleetDispatcher:
    """Pools requests across coroutines and dispatches via batch or sync.

    Currently supports Anthropic. The client must expose:
        client.messages.create(...)            # sync call
        client.messages.batches.create(...)    # batch submission
        client.messages.batches.retrieve(id)   # status
        client.messages.batches.results(id)    # iterator over results

    For testing, pass a fake client implementing the same surface.

    Example:
        async with FleetDispatcher(client, policy=RoutingPolicy.cost_aware()) as fleet:
            response = await fleet.submit(model="...", messages=[...])
    """

    def __init__(
        self,
        client: Any,
        policy: Optional[RoutingPolicy] = None,
        on_batch_submitted: Optional[Callable[[PendingBatch], None]] = None,
    ):
        self.client = client
        self.policy = policy or RoutingPolicy()
        self.stats = DispatchStats()
        self.on_batch_submitted = on_batch_submitted
        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._flusher_task: Optional[asyncio.Task] = None
        self._closed = False

    # --- lifecycle ----------------------------------------------------------

    async def __aenter__(self) -> FleetDispatcher:
        self._flusher_task = asyncio.create_task(self._flush_loop())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._flusher_task:
            # Push a shutdown sentinel; the flusher drains any pending work
            # and exits cleanly. No item ever gets orphaned.
            await self._queue.put(_SHUTDOWN)
            with contextlib.suppress(asyncio.CancelledError):
                await self._flusher_task

    # --- public API ---------------------------------------------------------

    async def submit(
        self,
        *,
        latency_budget_ms: Optional[int] = None,
        force_sync: bool = False,
        force_batch: bool = False,
        **params: Any,
    ) -> Any:
        """Submit a `messages.create` call. Routes based on policy.

        If `latency_budget_ms` is set and tighter than `sync_max_latency_ms`,
        runs synchronously. Otherwise queues for batch.
        """
        if self._closed:
            raise RuntimeError("FleetDispatcher closed")

        if force_sync or (
            not force_batch
            and latency_budget_ms is not None
            and latency_budget_ms <= self.policy.sync_max_latency_ms
        ):
            return await self._sync_call(params)

        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self._queue.put(_PendingItem(request_id, params, fut))
        return await fut

    async def submit_sync(self, **params: Any) -> Any:
        return await self.submit(force_sync=True, **params)

    async def submit_batch(self, **params: Any) -> Any:
        return await self.submit(force_batch=True, **params)

    # --- internals ----------------------------------------------------------

    async def _sync_call(self, params: dict) -> Any:
        try:
            result = await _maybe_await(self.client.messages.create, **params)
            self.stats.sync_calls += 1
            return result
        except Exception:
            self.stats.errors += 1
            raise

    async def _flush_loop(self) -> None:
        shutdown_pending = False
        while True:
            try:
                items, shutdown_seen = await self._collect_batch()
            except Exception:
                self.stats.errors += 1
                continue
            if shutdown_seen:
                shutdown_pending = True
            if items:
                try:
                    await self._submit_batch(items)
                except Exception:
                    self.stats.errors += 1
                    for it in items:
                        if not it.future.done():
                            it.future.set_exception(RuntimeError("flush error"))
            if shutdown_pending:
                # Final drain of anything that may still be in the queue.
                trailing: list[_PendingItem] = []
                while not self._queue.empty():
                    nxt = self._queue.get_nowait()
                    if nxt is _SHUTDOWN:
                        continue
                    trailing.append(nxt)
                for chunk in _chunks(trailing, self.policy.batch_max_size):
                    if chunk:
                        try:
                            await self._submit_batch(chunk)
                        except Exception:
                            self.stats.errors += 1
                return

    async def _collect_batch(self) -> tuple[list[_PendingItem], bool]:
        """Collect up to batch_max_size items or until window expires.

        Returns (items, shutdown_seen). `shutdown_seen` is True if we pulled
        the shutdown sentinel; the caller should drain and exit.
        """
        items: list[_PendingItem] = []
        deadline = time.time() + self.policy.batch_window_ms / 1000.0
        first = await self._queue.get()
        if first is _SHUTDOWN:
            return items, True
        items.append(first)

        while len(items) < self.policy.batch_max_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if nxt is _SHUTDOWN:
                return items, True
            items.append(nxt)

        return items, False

    async def _submit_batch(self, items: list[_PendingItem]) -> None:
        if not items:
            return
        request_objs = [{"custom_id": it.request_id, "params": it.params} for it in items]
        try:
            batch = await _maybe_await(
                self.client.messages.batches.create, requests=request_objs
            )
        except Exception as e:
            for it in items:
                if not it.future.done():
                    it.future.set_exception(e)
            self.stats.errors += 1
            return

        batch_id = getattr(batch, "id", None) or batch["id"]
        self.stats.batches_submitted += 1
        if self.on_batch_submitted:
            try:
                self.on_batch_submitted(
                    PendingBatch(batch_id=batch_id, request_count=len(items), submitted_at=time.time())
                )
            except Exception:
                pass

        # Poll for completion
        while True:
            await asyncio.sleep(self.policy.poll_interval_s)
            status = await _maybe_await(self.client.messages.batches.retrieve, batch_id)
            ps = getattr(status, "processing_status", None) or (
                status.get("processing_status") if isinstance(status, dict) else None
            )
            if ps == "ended":
                break

        # Dispatch results
        results_iter = self.client.messages.batches.results(batch_id)
        if hasattr(results_iter, "__aiter__"):
            async for result in results_iter:
                self._dispatch_result(items, result)
        else:
            for result in results_iter:
                self._dispatch_result(items, result)

        self.stats.batched_calls += len(items)

        # Any item without a result: error.
        for it in items:
            if not it.future.done():
                it.future.set_exception(
                    RuntimeError(f"No result returned for request {it.request_id}")
                )

    def _dispatch_result(self, items: list[_PendingItem], result: Any) -> None:
        rid = getattr(result, "custom_id", None) or (
            result.get("custom_id") if isinstance(result, dict) else None
        )
        if rid is None:
            return
        for it in items:
            if it.request_id == rid and not it.future.done():
                err = getattr(result, "error", None) or (
                    result.get("error") if isinstance(result, dict) else None
                )
                if err:
                    it.future.set_exception(RuntimeError(str(err)))
                else:
                    payload = getattr(result, "result", None) or (
                        result.get("result") if isinstance(result, dict) else result
                    )
                    it.future.set_result(payload)
                return


def _chunks(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


async def _maybe_await(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call `fn`; if it returned a coroutine, await it. Handles sync, async,
    and callable-instances-with-async-__call__ uniformly."""
    result = fn(*args, **kwargs)
    if inspect.iscoroutine(result) or inspect.isawaitable(result):
        return await result
    return result
