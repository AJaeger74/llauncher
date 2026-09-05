#!/usr/bin/env python3
"""Measure the VBR lazy-mode pool floor for one or more models.

The floor is the resident KV-pool VRAM a VBR server holds BEFORE any request
is sent (VBR allocates lazily). It is measured as:

    pool_floor = stable_process_VRAM - weights - mmproj

A clean floor requires: single model, NO --spec-type, NO --ctx-checkpoints,
NO restored context, ZERO completion requests.

Usage:
    python3 measure_floor.py <model.gguf> [more models...] \
        [--port 8090] [--ctx 262144] [--binary PATH] [--libdir PATH]

Models are measured one at a time (launch, settle, sample, stop) so VRAM is
freed between them. For BIG models the 27B :8080 "me" server must be STOPPED
first (see PLAN.md for the exact stop/run/restart batch). Small models (e.g.
the 1.7B) can be measured while :8080 is still up, since they fit in the free
VRAM.

Results are appended to /tmp/floor_results.json so a later session (after the
model is restarted) can read them and re-fit _vbr_lazy_base_mb().
"""
import sys, os, re, json, time, subprocess, signal, urllib.request

LLAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LLAUNCHER_DIR)

# Readable vbr-capable buun build (works as any user with the right libdir).
DEFAULT_BIN = "/opt/fast/ai/llama/buun-llama-cpp/build/bin/llama-server"
DEFAULT_LIBDIR = "/opt/fast/ai/llama/buun-llama-cpp/build/bin"
RESULTS = "/tmp/floor_results.json"


def gpu_apps():
    """[(pid, used_mib), ...] for every compute app."""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    apps = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        m = re.search(r"(\d+)", parts[1])
        if m:
            apps.append((int(parts[0].strip()), int(m.group(1))))
    return apps


def port_pid(port):
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port} " in line or f":{port}\t" in line or line.rstrip().endswith(f":{port}"):
            for pid in re.findall(r"pid=(\d+)", line):
                return int(pid)
    return None


def free_port(port):
    pid = port_pid(port)
    if not pid:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except Exception:
            return
        time.sleep(3)


def wait_health(port, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if d.read() == b'{"status":"ok"}':
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def measure(model, port, ctx, binary, libdir):
    from gguf_utils import get_model_info
    mi = get_model_info(model)
    weights_mib = mi.get("tensor_bytes", 0) / 1024 ** 2
    kv_layers = mi.get("kv_layer_count") or mi.get("block_count")
    arch = mi.get("arch")
    print(f"\n=== {os.path.basename(model)} ===", flush=True)
    print(f"arch={arch}  kv_layers={kv_layers}  weights={weights_mib:.0f} MiB", flush=True)

    free_port(port)
    env = dict(os.environ)
    if libdir:
        env["LD_LIBRARY_PATH"] = libdir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    log = f"/tmp/floor_srv_{port}.log"
    lf = open(log, "w")
    cmd = [binary, "-m", model, "-c", str(ctx),
           "--cache-type-k", "vbr", "--cache-type-v", "vbr",
           "-np", "1", "-t", "32", "-b", "128", "-ngl", "999",
           "--flash-attn", "on", "--port", str(port), "--host", "127.0.0.1"]
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_health(port):
        print("!! server not healthy — log:", log, flush=True)
        proc.terminate()
        return None

    time.sleep(5)  # let the VBR pool settle; send NO request
    pid = port_pid(port) or proc.pid
    samples = []
    for _ in range(3):
        for p, mib in gpu_apps():
            if p == pid:
                samples.append(mib)
        time.sleep(2)

    # stop the server, free VRAM for the next model / for :8080 to return
    proc.terminate()
    time.sleep(4)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass

    if not samples:
        print("!! no VRAM sample captured for pid", pid, flush=True)
        return None

    total = max(samples)  # stable while idle; max guards against a mid-free dip
    pool = total - weights_mib  # no mmproj loaded in this protocol
    rec = {
        "model": model, "arch": arch, "kv_layers": kv_layers,
        "weights_mib": round(weights_mib, 1), "total_mib": total,
        "pool_mib": round(pool, 1), "ctx": ctx,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"stable total={total} MiB  ->  pool_floor ~= {pool:.0f} MiB", flush=True)
    return rec


def main():
    args = sys.argv[1:]
    port, ctx, binary, libdir = 8090, 262144, DEFAULT_BIN, DEFAULT_LIBDIR
    models = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--port":
            port, i = int(args[i + 1]), i + 2
        elif a == "--ctx":
            ctx, i = int(args[i + 1]), i + 2
        elif a == "--binary":
            binary, i = args[i + 1], i + 2
        elif a == "--libdir":
            libdir, i = args[i + 1], i + 2
        else:
            models.append(a); i += 1
    if not models:
        print(__doc__)
        sys.exit(1)

    results = []
    try:
        results = json.load(open(RESULTS))
    except Exception:
        pass
    for m in models:
        r = measure(m, port, ctx, binary, libdir)
        if r:
            results.append(r)
    json.dump(results, open(RESULTS, "w"), indent=2)
    print(f"\n{len(results)} points in {RESULTS}:")
    for r in results:
        print(f"  {r['kv_layers']:>3} KV  {str(r['arch']):<16} pool~={r['pool_mib']} MiB  "
              f"{os.path.basename(r['model'])}")


if __name__ == "__main__":
    main()
