# Boxes on the CPU: BoxTracker

`BoxTracker` finds and follows objects on many cameras with **YOLOX-Nano on the CPU**, through OpenVINO. It is for edge boxes with no NVIDIA GPU, such as an Intel N100 mini PC. It gives boxes, not outlines, and it needs neither PyTorch nor SAM: OpenVINO and NumPy only, so `import tamserve.boxes` does not load torch.

It serves the way the rest of tamserve does:

- **One copy of the model for every camera.** The cameras that look on a given round go through YOLOX together, in one batch.
- **Looks are spread out.** YOLOX looks at each camera every `look_every` frames, and no more cameras look in one round than those intervals call for on average.
- **Cheap between looks.** A Kalman filter per object carries each box forward on the frames no look sees, for microseconds each.
- **ByteTrack's matching.** Confident finds are matched first, weak finds may only confirm an object already followed, and a new object must be seen by a second look before it is shown (except on a camera's first look).
- **Gray in, one channel.** Frames go in as the gray picture itself: a camera's gray stream, or the Y plane of its colour stream.

There is also a Rust port, [`rust/`](../rust/README.md): a single binary that gives the same ids and boxes on the same frames, with about half the memory overhead and no Python.

## Install

On a machine with PyTorch already, `pip install "tamserve[cpu]"`. On an edge box, leave PyTorch out:

```bash
pip install --no-deps tamserve
pip install "openvino>=2024.0" numpy pillow opencv-python-headless
```

Making the 1-channel model needs `onnx` (`pip install onnx`), once, on any machine.

## Make the 1-channel model

YOLOX's released ONNX files take 3 channels and a batch of 1. `gray_model()` turns one into a model that takes 1 channel and any batch:

```python
from tamserve.boxes import gray_model

# https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
gray_model("yolox_nano.onnx", "yolox_nano_gray.onnx")
```

This is exact: it changes nothing the model computes on a gray picture. A gray picture fed to a 3-channel model has the same image `g` in every channel, so the first convolution computes `Σ_c W_c * g = (Σ_c W_c) * g`, and summing its weights over the colour channels gives a convolution that reads `g` once. YOLOX starts with a Focus layer that cuts the picture into 4 interleaved patches and stacks them, 3 channels each, so its first convolution reads 12 channels; they are summed in groups of 3, to 4. On YOLOX-Nano the two models agree to 1e-4 in float32, which is float rounding. The identity, in Lean 4 with Mathlib (per pixel; the convolution is a sum of these):

```lean
theorem fold_gray {n : ℕ} (W : Fin n → ℝ) (g : ℝ) :
    ∑ c, W c * g = (∑ c, W c) * g := by
  rw [Finset.sum_mul]
```

Feeding a gray picture to a model trained on colour pictures does cost some accuracy, whichever way it is fed. Fine-tuning YOLOX on gray frames from your own cameras gets that back.

## Use it

```python
from tamserve.boxes import BoxTracker, Detector

detector = Detector("yolox_nano_gray.onnx", threads=3, classes=("person", "car", "truck"))
tracker = BoxTracker(detector, look_every=3)

while True:
    frames = {name: camera.take() for name, camera in cameras.items()}  # H x W uint8 gray, the newest of each
    found = tracker.step(frames)
    # found["gate"] is {1: Found(box=(left, top, right, bottom), score=0.83, label="person"), ...}
```

Each object gets a number, counted from 1 on each camera, and keeps it while it is followed. Someone a look no longer sees is left out of the answer but still followed for `hold` frames, and has their number back if a look sees them again. `tracker.drop("gate")` forgets a camera.

Frames can also be H x W x 3 RGB arrays; they are converted to gray with BT.601's weights. The cheapest gray picture is the one the decoder already made: the Y plane of a YUV frame, which needs no conversion at all.

| setting | what it does |
|---|---|
| `Detector(..., threads=3)` | CPU threads for OpenVINO. Leave a core for decoding and tracking: 3 on a 4-core N100. Below full load, fewer threads use less CPU: see below. |
| `Detector(..., pin=True)` | keeps OpenVINO's threads on their cores, so decoders do not push them about |
| `Detector(..., classes=("person",))` | keeps only these classes |
| `Detector(..., size=320)` | a smaller square for YOLOX to look at: cheaper, and it misses more of what is small. 416 is what YOLOX-Nano was trained at. |
| `Detector(..., threshold=0.1)` | finds below this confidence are ignored. Weak finds above it can still confirm objects. |
| `Detector(..., precision="f32")` | OpenVINO picks float32 on an N100 and bfloat16 on CPUs that have it, which is faster and moves scores by about a hundredth |
| `BoxTracker(..., look_every=3)` | frames between a camera's looks. A dict gives cameras their own, such as `{"gate": 1}`; the others keep 3. It can be changed while running. |
| `BoxTracker(..., hold=30)` | frames an object no look sees is still followed, in case it comes back |
| `BoxTracker(..., start=0.6, high=0.5)` | the confidence to start a new object, and to be matched in the first round |

## CPU and memory

`bench/boxes_cpu.py` runs the same cameras two ways: all in one process with one shared model, and one process per camera with a model of its own. Cameras are paced at a frame rate; frames are made up front, so decoding is left out. These were measured on a **2-core cloud VM** (Intel Xeon, OpenVINO 2026.4, float32, `look_every=3`, 720p), not on an N100, so take the shape of the results rather than the numbers:

| cameras | frames/s each | setup | CPU (cores) | memory | frames/s each, kept |
|---|---|---|---|---|---|
| 4 | 10 | one process, 1 thread | 0.24 | 304 MB | 10.0 |
| 4 | 10 | 4 processes | 0.26 | 802 MB | 10.0 |
| 8 | 25 | one process, 1 thread | **0.96** | **439 MB** | **24.9** |
| 8 | 25 | one process, 2 threads | 1.22 | 463 MB | 25.0 |
| 8 | 25 | 8 processes | 1.06 | 1,602 MB | 23.4 |

Both setups hold the same frames in memory (about 27 MB a camera, made by the benchmark), so the gap is all overhead: each process carries its own Python, OpenVINO and compiled model, about 175 MB, where one shared process adds a few MB a camera.

- **Memory falls the most:** 2.6 to 3.6 times less here, and the gap grows with every camera.
- **CPU falls a little**, about a tenth at 8 cameras, and the shared process kept every camera at its frame rate where separate processes, fighting over the cores, fell behind.
- **Batching does little for the CPU itself.** YOLOX-Nano is small enough that one frame already keeps a core busy, so a batch costs about what its frames would cost one by one. That is why batches are not padded to fixed sizes: the padding was wasted work.
- **More threads are not free.** A second OpenVINO thread used a quarter more CPU for the same frames. Add threads when a single one cannot keep up, not before.

On a real edge box the largest cost is often decoding, which this benchmark leaves out. Detect on the camera's low-resolution sub-stream, decode the full colour frame only for an alert, and let the integrated GPU's hardware decoder do the decoding if you can: that frees the CPU for detection without moving detection off it.

```bash
python bench/boxes_cpu.py --model yolox_nano_gray.onnx --cameras 4,8 --fps 10 --seconds 30 --threads 1 --image some.jpg
```

## INT8

`quantize(model, frames, dst)` quantises the model to INT8 with NNCF (`pip install nncf`), calibrated on frames from your cameras. **Check it before you deploy it.** On the VM above, calibrated on synthetic frames, INT8 kept only about 6 in 10 of float32's finds and lost the dog in YOLOX's own test picture, and it was not faster, which may be that VM: the N100 has AVX-VNNI, which OpenVINO's INT8 kernels use. YOLOX-Nano is small and it is reading gray pictures it was not trained on, and both make post-training quantisation harder. Calibrate on a few hundred real frames, compare against float32 with MCC on the decisions your rules make (below), and if INT8 still loses too much, fine-tune on gray frames first.

## Scoring

Box-level scores need ground truth boxes. For a camera's rules what matters is the decision, such as "a person is in the exclusion zone", per camera per frame. Scored that way there are true negatives, so the Matthews correlation coefficient is defined and is the fairest single number when most frames have nothing in them. `notebooks/cpu_boxes.ipynb` computes it.

## Limits

- **Boxes only.** For outlines, use `PeopleTracker` or `Tracker` on an NVIDIA GPU.
- **A box per class.** Objects are matched within their class, so a find that changes class between looks starts a new object.
- **The matching is greedy**, best overlap first, as `Tracker` matches; ByteTrack itself uses the Hungarian algorithm, which can differ when objects overlap.
- **Not through a Router yet**, which runs SAM 2 engines.
