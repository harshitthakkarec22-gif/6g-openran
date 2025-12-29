#!/usr/bin/env python3
"""Software-only OpenRAN pipeline simulation for x86_64 laptops/desktops.

Designed for the provided hardware profile:
- Architecture: x86_64
- CPU: Intel(R) Core(TM) i5-1035G1, 8 logical CPUs

The script simulates a CU -> DU -> RU processing pipeline, prints hardware
information, and reports throughput/latency metrics. It uses only the Python
standard library and requires no RF hardware.
"""

import argparse
import asyncio
import math
import os
import platform
import random
import statistics
import time
from typing import Any, Dict

MIN_PROCESS_DELAY_SEC = 0.0005  # 0.5 ms floor to avoid busy-loop when jitter is tiny
MIN_ELAPSED_TIME_SEC = 1e-9  # guard for throughput division
P99_MIN_SAMPLES = 10  # avoid noisy percentile estimates on tiny samples
MIN_DELAY_MS = 0.1  # lower bound for per-stage processing time in ms
MIN_STAGE_FRACTION = 0.25  # clamp per-stage delay to a fraction of its mean
MIN_FRAME_SIZE = 800
MAX_FRAME_SIZE = 1500


def percentile(values: list[float], percent: float) -> float:
    """Simple percentile helper to avoid heavy quantile computation."""
    if not values:
        return 0.0
    bounded_percent = min(max(percent, 0.0), 100.0)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (bounded_percent / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def describe_hardware() -> Dict[str, Any]:
    """Return and print a brief hardware summary."""
    summary = {
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "python_version": platform.python_version(),
    }

    print("=== Host Hardware Summary ===")
    print(f"Architecture : {summary['architecture']}")
    print(f"CPU Model    : {summary['processor']}")
    print(f"Logical CPUs : {summary['logical_cpus']}")
    print(f"Python       : {summary['python_version']}")
    print("================================\n")
    return summary


class Metrics:
    def __init__(self) -> None:
        self.processed = 0
        self.bytes = 0
        self.latencies = []  # seconds

    def record(self, frame: Dict[str, Any]) -> None:
        self.processed += 1
        self.bytes += frame["size"]
        self.latencies.append(time.time() - frame["created_at"])

    def snapshot(self, start_time: float) -> Dict[str, float]:
        elapsed = max(time.time() - start_time, MIN_ELAPSED_TIME_SEC)
        latency_ms = [v * 1000 for v in self.latencies]
        p99 = percentile(latency_ms, 99) if len(latency_ms) >= P99_MIN_SAMPLES else 0.0
        return {
            "processed": self.processed,
            "throughput_mbps": (self.bytes * 8) / (1_000_000 * elapsed),
            "latency_mean_ms": statistics.mean(latency_ms) if latency_ms else 0.0,
            "latency_p99_ms": p99,
            "elapsed": elapsed,
        }


async def traffic_source(out_q: asyncio.Queue, rate: float, runtime: int, ue_count: int) -> None:
    """Generate traffic frames for the simulation."""
    if rate <= 0:
        # No traffic generation requested; wait out the runtime window.
        await asyncio.sleep(runtime)
        await out_q.put(None)
        return
    period = 1.0 / rate
    end_time = time.time() + runtime
    frame_id = 0
    while time.time() < end_time:
        frame = {
            "id": frame_id,
            "ue": frame_id % ue_count,
            "size": random.randint(MIN_FRAME_SIZE, MAX_FRAME_SIZE),
            "created_at": time.time(),
        }
        await out_q.put(frame)
        frame_id += 1
        await asyncio.sleep(period)
    await out_q.put(None)


async def component(in_q: asyncio.Queue, out_q: asyncio.Queue, mean_ms: float, jitter_ms: float) -> None:
    """Process frames and pass them downstream."""
    while True:
        frame = await in_q.get()
        if frame is None:
            await out_q.put(None)
            in_q.task_done()
            break
        raw_ms = random.gauss(mean_ms, jitter_ms)
        bounded_raw = max(raw_ms, mean_ms * MIN_STAGE_FRACTION)
        delay_ms = max(MIN_DELAY_MS, bounded_raw)
        delay = max(MIN_PROCESS_DELAY_SEC, delay_ms / 1000)
        await asyncio.sleep(delay)
        await out_q.put(frame)
        in_q.task_done()


async def sink(in_q: asyncio.Queue, metrics: Metrics, done: asyncio.Event) -> None:
    """Collect frames and update metrics."""
    while True:
        frame = await in_q.get()
        if frame is None:
            in_q.task_done()
            done.set()
            break
        metrics.record(frame)
        in_q.task_done()


async def reporter(metrics: Metrics, start_time: float, done: asyncio.Event, interval: int) -> None:
    """Print periodic metrics until the run completes."""
    while not done.is_set():
        await asyncio.sleep(interval)
        snap = metrics.snapshot(start_time)
        print(
            f"[t={snap['elapsed']:.1f}s] processed={snap['processed']}, "
            f"throughput={snap['throughput_mbps']:.2f} Mbps, "
            f"latency_mean={snap['latency_mean_ms']:.2f} ms, "
            f"latency_p99={snap['latency_p99_ms']:.2f} ms"
        )


async def run_simulation(args: argparse.Namespace) -> None:
    describe_hardware()
    if args.rate < 0:
        raise ValueError("rate must be non-negative")
    if args.ue_count <= 0:
        raise ValueError("ue-count must be positive")
    for field in ("cu_ms", "du_ms", "ru_ms", "jitter_ms"):
        if getattr(args, field) < 0:
            raise ValueError(f"{field.replace('_', '-')} must be non-negative")
    start_time = time.time()
    metrics = Metrics()
    done = asyncio.Event()

    q_source = asyncio.Queue(maxsize=args.buffer)
    q_cu = asyncio.Queue(maxsize=args.buffer)
    q_du = asyncio.Queue(maxsize=args.buffer)
    q_ru = asyncio.Queue(maxsize=args.buffer)

    tasks = [
        asyncio.create_task(traffic_source(q_source, args.rate, args.runtime, args.ue_count)),
        asyncio.create_task(component(q_source, q_cu, args.cu_ms, args.jitter_ms)),
        asyncio.create_task(component(q_cu, q_du, args.du_ms, args.jitter_ms)),
        asyncio.create_task(component(q_du, q_ru, args.ru_ms, args.jitter_ms)),
        asyncio.create_task(sink(q_ru, metrics, done)),
        asyncio.create_task(reporter(metrics, start_time, done, args.report_interval)),
    ]

    await asyncio.gather(*tasks)
    snap = metrics.snapshot(start_time)
    print("\n=== Simulation Complete ===")
    print(f"Duration     : {snap['elapsed']:.2f} s")
    print(f"Frames       : {snap['processed']}")
    print(f"Throughput   : {snap['throughput_mbps']:.2f} Mbps")
    print(f"Latency mean : {snap['latency_mean_ms']:.2f} ms")
    print(f"Latency p99  : {snap['latency_p99_ms']:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Software-only OpenRAN pipeline simulator")
    parser.add_argument("--runtime", type=int, default=10, help="Simulation duration in seconds")
    parser.add_argument("--ue-count", type=int, default=4, help="Number of simulated UEs")
    parser.add_argument("--rate", type=float, default=15.0, help="Packets per second generated by the source")
    parser.add_argument("--buffer", type=int, default=128, help="Queue size between components")
    parser.add_argument("--cu-ms", type=float, default=3.0, help="Average CU processing time in ms")
    parser.add_argument("--du-ms", type=float, default=4.0, help="Average DU processing time in ms")
    parser.add_argument("--ru-ms", type=float, default=2.5, help="Average RU processing time in ms")
    parser.add_argument("--jitter-ms", type=float, default=1.0, help="Standard deviation for processing time in ms")
    parser.add_argument("--report-interval", type=int, default=2, help="Seconds between metric reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_simulation(args))


if __name__ == "__main__":
    main()
