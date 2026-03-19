"""
Resource keeper to prevent JupyterHub idle culling.
Auto-detects GPU/CPU/RAM and maintains: GPU >40%, CPU >20%, Memory >20%
"""

import multiprocessing
import threading
import time
import os
import signal
import sys

# --- Target utilization fractions ---
GPU_MEM_FRACTION = 0.7   
CPU_FRACTION = 0.7        
MEM_FRACTION = 0.7       


def detect_gpus():
    """Auto-detect available NVIDIA GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'id': i,
                'name': props.name,
                'total_mem_gb': props.total_memory / 1e9,
            })
        return gpus
    except ImportError:
        return []


def detect_cpu():
    """Auto-detect CPU core count."""
    return os.cpu_count() or 1


def detect_mem_gb():
    """Auto-detect total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # fallback: read from /proc/meminfo
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    return 16  # safe fallback


def gpu_worker(gpu_id, mem_fraction):
    """Occupy GPU memory and run a simple kernel to keep utilization up."""
    try:
        import torch
        torch.cuda.set_device(gpu_id)
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
        alloc_bytes = int(total_mem * mem_fraction)
        alloc_elements = alloc_bytes // 4  # float32

        buf = torch.randn(alloc_elements, device=f'cuda:{gpu_id}')
        print(f"[GPU {gpu_id}] Allocated {alloc_bytes / 1e9:.1f} GB ({mem_fraction*100:.0f}%)")

        a = torch.randn(1024, 1024, device=f'cuda:{gpu_id}')
        while True:
            _ = torch.mm(a, a)
            time.sleep(0.5)
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")


def cpu_worker():
    """Burn CPU cycles."""
    x = 0.0
    while True:
        for _ in range(1_000_000):
            x += 0.0001
            x *= 0.9999
        time.sleep(0.001)


def mem_worker(gb):
    """Allocate and touch memory to keep it resident."""
    size = int(gb * 1024 ** 3)
    try:
        buf = bytearray(size)
        print(f"[MEM] Allocated {gb:.1f} GB")
        page_size = 4096
        while True:
            for offset in range(0, len(buf), page_size * 256):
                buf[offset] = buf[offset] ^ 0xFF
                buf[offset] = buf[offset] ^ 0xFF
            time.sleep(10)
    except MemoryError:
        half = gb / 2
        print(f"[MEM] Could not allocate {gb:.1f} GB, trying {half:.1f} GB")
        buf = bytearray(int(half * 1024 ** 3))
        while True:
            time.sleep(60)


def main():
    # --- Auto-detect resources ---
    gpus = detect_gpus()
    total_cores = detect_cpu()
    total_mem_gb = detect_mem_gb()

    target_cores = max(1, int(total_cores * CPU_FRACTION))
    target_mem_gb = int(total_mem_gb * MEM_FRACTION)

    print("=== Keep-Alive Resource Holder ===")
    print(f"[Detected] CPU: {total_cores} cores | RAM: {total_mem_gb:.0f} GB | GPUs: {len(gpus)}")
    if gpus:
        for g in gpus:
            print(f"  GPU {g['id']}: {g['name']} ({g['total_mem_gb']:.0f} GB)")
    print(f"[Target]   CPU: {target_cores} cores ({CPU_FRACTION*100:.0f}%) | "
          f"RAM: {target_mem_gb} GB ({MEM_FRACTION*100:.0f}%) | "
          f"GPU mem: {GPU_MEM_FRACTION*100:.0f}%")
    print("Press Ctrl+C to stop.\n")

    threads = []

    # GPU threads
    for g in gpus:
        t = threading.Thread(target=gpu_worker, args=(g['id'], GPU_MEM_FRACTION), daemon=True)
        t.start()
        threads.append(t)

    # Memory thread
    t = threading.Thread(target=mem_worker, args=(target_mem_gb,), daemon=True)
    t.start()
    threads.append(t)

    # CPU workers
    cpu_procs = []
    for _ in range(target_cores):
        p = multiprocessing.Process(target=cpu_worker, daemon=True)
        p.start()
        cpu_procs.append(p)

    def cleanup(sig=None, frame=None):
        print("\n[STOP] Cleaning up...")
        for p in cpu_procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
