"""Find and follow boxes on many cameras with YOLOX-Nano on the CPU, through OpenVINO.

The CPU counterpart of PeopleTracker, for edge boxes with no NVIDIA GPU. It follows
boxes, not outlines, and it needs neither PyTorch nor SAM: OpenVINO and NumPy only,
so `import tamserve.boxes` does not load torch.

It keeps tamserve's way of serving: one copy of the model for every camera, and
whatever needs it at a given moment goes through it in one batch. YOLOX looks at
each camera every `look_every` frames, the looks are spread so that they do not all
land in the same round, and between looks a Kalman filter per object carries each
box forward, which costs microseconds. Objects are matched to what a look finds as
ByteTrack matches them: confident finds first, then weak finds may only confirm an
object already followed.

The model takes one channel: frames go in as the gray picture itself (a camera's
gray stream, or the Y plane of its colour stream). gray_model() turns YOLOX's own
3-channel ONNX file into one, exactly: for a picture whose three channels are the
same gray image g, the first convolution gives sum_c W_c * g = (sum_c W_c) * g, so
summing its weights over the colour channels changes nothing it computes.
docs/cpu-boxes.md has the Lean 4 statement of that identity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush")
_PAD = 114  # the grey YOLOX pads its letterbox with
_STRIDES = (8, 16, 32)


class Found(NamedTuple):
    """One followed object on one frame."""
    box: tuple    # (left, top, right, bottom) in the frame's own pixels
    score: float  # the confidence of the latest find that matched it
    label: str    # its class


# --- the model ------------------------------------------------------------------------------------------------

def gray_model(src, dst) -> Path:
    """YOLOX's ONNX file (3 channels, batch 1) as one that takes 1 channel and any batch.

    The first convolution's weights are summed over the colour channels. YOLOX starts
    with a Focus layer, which cuts the picture into 4 interleaved patches and stacks
    them, 3 channels each, so that convolution reads 12 channels: they are summed in
    groups of 3, to 4. Reshapes that name batch 1 are made to keep the input's batch.
    Needs the `onnx` package; the result runs with OpenVINO alone."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    (image,) = [i for i in g.input if i.name not in inits]
    dims = image.type.tensor_type.shape.dim
    if dims[1].dim_value != 3:
        raise ValueError(f"expected a 3-channel YOLOX model, not {dims[1].dim_value} channels")
    conv = next(n for n in g.node if n.op_type == "Conv")
    w = numpy_helper.to_array(inits[conv.input[1]])
    if w.shape[1] % 3:
        raise ValueError(f"the first convolution reads {w.shape[1]} channels, not a multiple of 3")
    folded = w.reshape(w.shape[0], w.shape[1] // 3, 3, *w.shape[2:]).sum(2, dtype=np.float64).astype(w.dtype)
    inits[conv.input[1]].CopyFrom(numpy_helper.from_array(folded, conv.input[1]))
    dims[0].ClearField("dim_value")
    dims[0].dim_param = "batch"
    dims[1].dim_value = 1
    for out in g.output:
        out.type.tensor_type.shape.dim[0].dim_param = "batch"
    for node in g.node:
        if node.op_type == "Reshape" and node.input[1] in inits:
            shape = numpy_helper.to_array(inits[node.input[1]]).copy()
            if shape[0] == 1:
                shape[0] = 0  # ONNX: 0 copies the input's own size, here its batch
                inits[node.input[1]].CopyFrom(numpy_helper.from_array(shape, node.input[1]))
    del g.value_info[:]  # shapes inferred for batch 1 no longer hold
    onnx.checker.check_model(m)
    dst = Path(dst)
    onnx.save(m, str(dst))
    return dst


def quantize(model, frames, dst, *, size: int = 416) -> Path:
    """`model` (ONNX or OpenVINO IR, 1-channel) quantised to INT8 with NNCF, calibrated
    on `frames`: a few hundred gray frames from the cameras it will watch. Saved as
    OpenVINO IR (dst.xml and dst.bin). Needs the `nncf` package."""
    import nncf
    import openvino as ov

    core = ov.Core()
    m = core.read_model(str(model))
    data = [_letterbox([f], size)[0] for f in (_gray(f) for f in frames)]
    q = nncf.quantize(m, nncf.Dataset(data), preset=nncf.QuantizationPreset.MIXED, subset_size=len(data))
    dst = Path(dst).with_suffix(".xml")
    ov.save_model(q, str(dst))
    return dst


def _gray(frame) -> np.ndarray:
    """An H x W uint8 gray frame; an H x W x 3 RGB one is converted as BT.601 weighs it."""
    a = np.asarray(frame)
    if a.dtype != np.uint8 or a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)) or 0 in a.shape:
        raise ValueError(f"a frame is an H x W uint8 gray array (or H x W x 3 RGB), not {a.dtype} {a.shape}")
    if a.ndim == 2:
        return a
    if a.shape[2] == 1:
        return a[:, :, 0]
    return (a @ np.array([0.299, 0.587, 0.114], dtype=np.float32) + 0.5).astype(np.uint8)


def _resize(gray, w, h):
    try:
        import cv2
        return cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)  # what YOLOX itself uses
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(gray).resize((w, h), Image.BILINEAR))


def _letterbox(grays, size, out=None):
    """Frames as YOLOX takes them: scaled to fit size x size, top left, padded with grey.
    Returns (n x 1 x size x size float32, each frame's scale)."""
    x = np.full((len(grays), 1, size, size), _PAD, np.float32) if out is None else out
    scales = []
    for i, g in enumerate(grays):
        r = min(size / g.shape[0], size / g.shape[1])
        h, w = max(1, round(g.shape[0] * r)), max(1, round(g.shape[1] * r))
        x[i, 0, :h, :w] = _resize(g, w, h)
        scales.append(r)
    return x, scales


def _grid(size):
    """Each anchor's cell (x, y) and stride, in the order YOLOX's head lists them."""
    cells, strides = [], []
    for s in _STRIDES:
        n = size // s
        ys, xs = np.mgrid[0:n, 0:n]
        cells.append(np.stack([xs.ravel(), ys.ravel()], 1))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(cells).astype(np.float32), np.concatenate(strides).astype(np.float32)


def _nms(boxes, scores, least):
    """Indices kept by greedy non-maximum suppression, the most confident first."""
    order, keep = np.argsort(-scores), []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        order = rest[_iou(boxes[i:i + 1], boxes[rest])[0] <= least]
    return np.array(keep, dtype=int)


def _iou(a, b) -> np.ndarray:
    """The overlap (IoU) of every box in a with every box in b, len(a) x len(b)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), 2)
    area = lambda x: np.prod(np.clip(x[:, 2:] - x[:, :2], 0, None), 1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


class Detector:
    """YOLOX on the CPU through OpenVINO, one call for a batch of frames from any cameras.

    model: a 1-channel YOLOX model, ONNX or OpenVINO IR (gray_model(), quantize()).
    size: the side of the square YOLOX looks at: 416 for YOLOX-Nano, as it was trained.
        320 is cheaper and misses more of what is small.
    threads: CPU threads OpenVINO may use. Leave a core for decoding and tracking:
        3 on a 4-core N100. None lets OpenVINO decide.
    pin: keep OpenVINO's threads on their cores, so the decoders do not push them about.
    threshold: the confidence below which a find is ignored.
    overlap: finds of one class that overlap a more confident one by more than this go.
    classes: the class names to keep, such as ("person",); all of them when left out.
    names: the model's class names, in order. COCO's for YOLOX's released models.
    precision: OpenVINO's inference precision, such as "f32". By default OpenVINO picks:
        float32 on an N100, bfloat16 on CPUs that have it, which is faster and moves
        scores by about a hundredth.
    """

    def __init__(self, model, *, size: int = 416, threads: int | None = None, pin: bool = True,
                 threshold: float = 0.1, overlap: float = 0.45, classes=None, names=COCO, device: str = "CPU",
                 precision: str | None = None):
        import openvino as ov

        self.size, self.threshold, self.overlap, self.names = size, threshold, overlap, tuple(names)
        unknown = set(classes or ()) - set(self.names)
        if unknown:
            raise ValueError(f"no such classes in this model: {sorted(unknown)}")
        self.keep = None if classes is None else np.array([self.names.index(c) for c in classes])
        core = ov.Core()
        m = core.read_model(str(model))
        if m.input(0).get_partial_shape()[1].get_length() != 1:
            raise ValueError("Detector takes a 1-channel model: make one with gray_model()")
        m.reshape([-1, 1, size, size])
        config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}
        if device == "CPU":
            config["ENABLE_CPU_PINNING"] = "YES" if pin else "NO"
            if threads:
                config["INFERENCE_NUM_THREADS"] = str(threads)
        if precision:
            config["INFERENCE_PRECISION_HINT"] = precision
        self.model = core.compile_model(m, device, config)
        self.request = self.model.create_infer_request()
        self.cells, self.strides = _grid(size)
        self._input: dict = {}  # batch size -> the array frames are letterboxed into (a few: as many as cameras look at once)

    def __call__(self, frames: list) -> list:
        """frames: H x W uint8 gray arrays (or RGB, converted). Returns, for each, the
        finds as (boxes k x 4 in the frame's pixels, scores k, class indices k)."""
        if not frames:
            return []
        grays = [_gray(f) for f in frames]
        n = len(grays)  # not padded: OpenVINO's CPU plugin takes any batch, and padding is wasted work
        buf = self._input.get(n)
        if buf is None:
            buf = self._input[n] = np.empty((n, 1, self.size, self.size), np.float32)
        buf.fill(_PAD)
        x, scales = _letterbox(grays, self.size, buf)
        raw = self.request.infer({0: x})[self.model.output(0)][:len(grays)]
        return [self._decode(r, s, g.shape) for r, s, g in zip(raw, scales, grays)]

    def _decode(self, raw, scale, shape):
        """One frame's raw head output (anchors x (4 + 1 + classes)) as finds."""
        cls = raw[:, 5:] if self.keep is None else raw[:, 5:][:, self.keep]
        best = cls.argmax(1)
        score = raw[:, 4] * cls[np.arange(len(cls)), best]
        hit = score > self.threshold
        if not hit.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        r, score, best = raw[hit], score[hit], best[hit]
        xy = (r[:, :2] + self.cells[hit]) * self.strides[hit]
        wh = np.exp(np.minimum(r[:, 2:4], 10)) * self.strides[hit]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / scale
        boxes = np.clip(boxes, 0, [shape[1], shape[0], shape[1], shape[0]])
        label = best if self.keep is None else self.keep[best]
        # One class's finds only suppress each other: shift each class to its own corner of the plane.
        kept = _nms(boxes + (label * 10_000.0)[:, None], score, self.overlap)
        return boxes[kept].astype(np.float32), score[kept].astype(np.float32), label[kept]


# --- following ------------------------------------------------------------------------------------------------

class _Kalman:
    """ByteTrack's filter: centre x, y, aspect ratio and height, and their velocities."""
    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)
    _POS, _VEL = 1 / 20, 1 / 160

    def __init__(self, box):
        z = self._measure(box)
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        self.P = np.diag(np.square([2 * self._POS * h, 2 * self._POS * h, 1e-2, 2 * self._POS * h,
                                    10 * self._VEL * h, 10 * self._VEL * h, 1e-5, 10 * self._VEL * h]))

    @staticmethod
    def _measure(box):
        w, h = box[2] - box[0], box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2, w / max(h, 1e-6), max(h, 1e-6)])

    def predict(self):
        h = self.x[3]
        q = np.square([self._POS * h, self._POS * h, 1e-2, self._POS * h,
                       self._VEL * h, self._VEL * h, 1e-5, self._VEL * h])
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + np.diag(q)

    def update(self, box):
        h = self.x[3]
        r = np.diag(np.square([self._POS * h, self._POS * h, 1e-1, self._POS * h]))
        s = self._H @ self.P @ self._H.T + r
        k = np.linalg.solve(s, self._H @ self.P).T
        self.x = self.x + k @ (self._measure(box) - self._H @ self.x)
        self.P = self.P - k @ self._H @ self.P

    def box(self):
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class _Object:
    kalman: _Kalman
    label: int
    score: float
    confirmed: bool
    unseen: int = 0  # frames since a look last saw it


@dataclass
class _Camera:
    objects: dict = field(default_factory=dict)  # id -> _Object
    since_look: int | None = None                # frames since the last look; None before the first
    next_id: int = 1
    size: tuple | None = None


def _match(tracks: list, boxes: np.ndarray, least: float) -> dict:
    """box index -> track index, one to one, the best overlap first, down to `least`."""
    if not tracks or not len(boxes):
        return {}
    iou = _iou(boxes, np.stack([t.kalman.box() for t in tracks]))
    out, taken = {}, set()
    for flat in np.argsort(-iou, axis=None):
        i, j = divmod(int(flat), len(tracks))
        if iou[i, j] < least:
            break
        if i not in out and j not in taken:
            out[i] = j
            taken.add(j)
    return out


def _due(cameras: dict, cams: list, every: dict) -> list:
    """Cameras that look this step, spread as tamserve.tracker._due spreads them: a new
    camera at once; others once `every` frames have passed, no more in one step than
    the intervals call for on average, those that look most often first."""
    new = [c for c in cams if cameras[c].since_look is None]
    due = [c for c in cams if (cameras[c].since_look or 0) >= every[c]]
    due.sort(key=lambda c: (every[c], -cameras[c].since_look))
    return new + due[:max(1, math.ceil(sum(1 / Fraction(k) for k in every.values())))]


class BoxTracker:
    """Find and follow objects on every camera, on the CPU; call step() once per round.

    detector: a Detector, or the path of a 1-channel model to build one from with
        `threads` and `classes`. Anything with a `names` tuple that can be called as
        a Detector is will do, such as another detector of your own.
    look_every: frames between a camera's looks: 1 looks at every frame, 3 costs a
        third as much and finds new things up to 2 frames later. A dict gives cameras
        their own, such as {"gate": 1}, and those it does not name keep 3. It can be
        changed while running.
    hold: frames an object no look has seen is still followed, out of the answer, so
        that it has its id back if a look sees it again.
    start: the confidence a find needs to start something new.
    high: finds from this confidence are matched first; weaker ones may only confirm
        an object already followed (ByteTrack's second round).
    """

    def __init__(self, detector, *, look_every: int | dict = 3, hold: int = 30, start: float = 0.6,
                 high: float = 0.5, threads: int | None = None, classes=None):
        for c in look_every if isinstance(look_every, dict) else [None]:
            self._interval(look_every, c)  # raises below 1
        if hold < 0:
            raise ValueError("hold must be at least 0")
        if isinstance(detector, (str, Path)):
            detector = Detector(detector, threads=threads, classes=classes)
        self.detector = detector
        self.look_every, self.hold, self.start, self.high = look_every, hold, start, high
        self.cameras: dict = {}
        self._joined = 0

    @staticmethod
    def _interval(look_every, cam, default=None):
        every = look_every.get(cam, default) if isinstance(look_every, dict) else look_every
        if every < 1:
            raise ValueError("look_every must be at least 1")
        return every

    def drop(self, cam) -> None:
        """Forget a camera."""
        self.cameras.pop(cam, None)

    def step(self, frames: dict) -> dict:
        """frames: camera id -> its newest frame, H x W uint8 gray (or H x W x 3 RGB).
        Returns camera id -> {object id: Found}, for what the latest look on that camera
        saw. Ids are whole numbers from 1, counted per camera."""
        for c, f in frames.items():
            shape = np.shape(f)[:2]
            cam = self.cameras.setdefault(c, _Camera())
            if cam.size not in (None, shape):
                self.drop(c)  # the camera changed resolution: start it afresh
                cam = self.cameras.setdefault(c, _Camera())
            cam.size = shape
            for o in cam.objects.values():
                o.kalman.predict()
                o.unseen += 1
        every = {c: self._interval(self.look_every, c, 3) for c in self.cameras}
        looking = _due(self.cameras, list(frames), every)
        for c, (boxes, scores, labels) in zip(looking, self.detector([frames[c] for c in looking])):
            self._look(self.cameras[c], boxes, scores, labels)
            cam = self.cameras[c]
            if cam.since_look is None:  # first look: set this camera's later looks apart from the others'
                cam.since_look = self._joined % every[c]
                self._joined += 1
            else:
                cam.since_look = 0
        out = {}
        for c in frames:
            cam = self.cameras[c]
            cam.since_look += 1
            out[c] = {t: Found(tuple(float(v) for v in o.kalman.box()), o.score, self.detector.names[o.label])
                      for t, o in cam.objects.items() if o.confirmed and o.unseen < cam.since_look}
        return out

    def _look(self, cam, boxes, scores, labels):
        """ByteTrack's association, per class: confident finds against everything
        followed, weak finds against what is left, new objects on trial last."""
        first = cam.since_look is None
        # Classes in order, so new objects are numbered the same way on every run (and as the Rust port does).
        for label in sorted(set(labels.tolist()) | {o.label for o in cam.objects.values()}):
            mine = labels == label
            b, s = boxes[mine], scores[mine]
            ids = [t for t, o in cam.objects.items() if o.label == label]
            sure = [t for t in ids if cam.objects[t].confirmed]
            trial = [t for t in ids if not cam.objects[t].confirmed]
            strong, weak = np.flatnonzero(s >= self.high), np.flatnonzero(s < self.high)
            seen: set = set()

            def take(tracks, idx, least):
                hits = _match([cam.objects[t] for t in tracks], b[idx], least)
                for i, j in hits.items():
                    o = cam.objects[tracks[j]]
                    o.kalman.update(b[idx[i]])
                    o.score, o.unseen = float(s[idx[i]]), 0
                    o.confirmed = True
                    seen.add(tracks[j])
                return np.array([idx[i] for i in range(len(idx)) if i not in hits], dtype=int)

            strong = take(sure, strong, 0.2)
            take([t for t in sure if t not in seen], weak, 0.5)
            strong = take(trial, strong, 0.3)
            for t in trial:
                if t not in seen:
                    del cam.objects[t]  # on trial and not seen again: a false find
            for i in strong:
                if s[i] >= self.start:
                    cam.objects[cam.next_id] = _Object(_Kalman(b[i]), label, float(s[i]), confirmed=first)
                    cam.next_id += 1
        for t in [t for t, o in cam.objects.items() if o.unseen > self.hold]:
            del cam.objects[t]
