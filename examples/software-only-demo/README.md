# Software-Only Demo (x86_64 Laptop/Desktop)

This example provides a lightweight, software-only OpenRAN pipeline simulation that runs on the provided hardware profile (Intel® Core™ i5-1035G1, 8 threads, x86_64). It is designed for classroom or university project demonstrations without any RF hardware.

## What It Does

- Simulates CU → DU → RU processing with configurable traffic rate and UE count
- Prints periodic throughput and latency stats
- Verifies the host CPU/architecture and surfaces them in the output
- Uses only the Python standard library (no extra dependencies)

## Prerequisites

- Python 3.8+ (standard library only)
- Host similar to: x86_64, 8 logical CPUs (Intel i5-1035G1 or comparable)

## Run the Demo

```bash
cd examples/software-only-demo
python simulate_openran.py --runtime 8 --ue-count 6 --rate 20
```

Key arguments:
- `--runtime` seconds to run (default: 10)
- `--ue-count` number of simulated UEs (default: 4)
- `--rate` packets per second produced by the source (default: 15)

## Expected Output

- Hardware summary (architecture, CPU count, CPU model if available)
- Periodic stats showing processed frames, mean/p99 latency, and throughput
- Final summary after the run completes

## Notes

- This is a pure software demo; it does not transmit RF samples.
- Suitable for laptops/desktops that match or exceed the provided CPU profile.
- If the machine differs (fewer cores or different architecture), the script will still run and note the difference.
