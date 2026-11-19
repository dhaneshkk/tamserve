"""CPU and memory for BoxTracker: every camera in one process with one shared model,
against one process per camera with a model of its own (the usual way).

Cameras are paced at --fps each. Frames are made up front (720p gray, a textured
background with --image pasted in and moving), so decoding is left out: what is
measured is detection and tracking. Reports, for each setup, the CPU it used (in
cores: CPU seconds over wall seconds, summed over its processes), its resident
memory (summed over its processes), and the frame rate each camera got.

  python bench/boxes_cpu.py --model nano_gray.onnx --cameras 4,8 --fps 10 --seconds 20
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np


def frames(image, n=30, h=720, w=1280, seed=0):
    rng = np.random.default_rng(seed)
    import cv2
    bg = cv2.GaussianBlur(rng.integers(60, 200, (h, w), dtype=np.uint8), (0, 0), 9)
    pic = None if image is None else cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    out = []
    for i in range(n):
        f = bg.copy()
        if pic is not None:
            x = (40 + 4 * i + 37 * seed) % (w - pic.shape[1])
            f[60:60 + pic.shape[0], x:x + pic.shape[1]] = pic
        out.append(f)
    return out


def run(model, cams, fps, seconds, threads, look_every, image, ready=None, report=None):
    """Track `cams` cameras in this process for `seconds`, paced at `fps` rounds a second."""
    from tamserve.boxes import BoxTracker
    tracker = BoxTracker(model, threads=threads, look_every=look_every)
    videos = {c: frames(image, seed=c) for c in cams}
    for c in cams:  # warm up: the first call prepares the model's shapes
        tracker.step({c: videos[c][0]})
    if ready is not None:
        ready.wait()
    start, rounds = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        tracker.step({c: videos[c][rounds % len(videos[c])] for c in cams})
        rounds += 1
        wait = start + rounds / fps - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
    if report is not None:
        report.put(rounds / (time.perf_counter() - start))
    return rounds / (time.perf_counter() - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a 1-channel YOLOX model (tamserve.boxes.gray_model)")
    ap.add_argument("--cameras", default="4,8")
    ap.add_argument("--fps", type=float, default=10, help="frames a second each camera sends")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--threads", type=int, default=None, help="OpenVINO threads for the shared process")
    ap.add_argument("--look-every", type=int, default=3)
    ap.add_argument("--image", default=None, help="a picture to move across the frames, so there is something to track")
    a = ap.parse_args()
    import psutil
    ctx = mp.get_context("spawn")
    for n in map(int, a.cameras.split(",")):
        row = {"cameras": n, "fps_asked": a.fps, "look_every": a.look_every}
        for setup in ("shared", "separate"):
            groups = [list(range(n))] if setup == "shared" else [[c] for c in range(n)]
            ready, report = ctx.Event(), ctx.Queue()
            procs = [ctx.Process(target=run, args=(a.model, g, a.fps, a.seconds,
                                                   a.threads if setup == "shared" else 1, a.look_every, a.image,
                                                   ready, report)) for g in groups]
            for p in procs:
                p.start()
            time.sleep(min(60, 5 + 2 * len(procs)))  # load and warm up
            ps = [psutil.Process(p.pid) for p in procs]
            rss = sum(p.memory_info().rss for p in ps)
            cpu0, t0 = sum(sum(p.cpu_times()[:2]) for p in ps), time.perf_counter()
            ready.set()
            time.sleep(a.seconds * 0.9)
            rss = max(rss, sum(p.memory_info().rss for p in ps))
            cpu = (sum(sum(p.cpu_times()[:2]) for p in ps) - cpu0) / (time.perf_counter() - t0)
            rates = [report.get() for _ in procs]
            for p in procs:
                p.join()
            row[setup] = {"processes": len(procs), "cpu_cores": round(cpu, 2), "memory_mb": round(rss / 2**20),
                          "fps_per_camera": round(min(rates), 1)}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
