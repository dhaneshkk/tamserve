"""BoxTracker and Detector: YOLOX on the CPU. None of these need PyTorch."""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tamserve.boxes import BoxTracker, Detector, gray_model

ov = pytest.importorskip("openvino")

_YOLOX = "https://github.com/Megvii-BaseDetection/YOLOX/"
NANO = (_YOLOX + "releases/download/0.1.1rc0/yolox_nano.onnx",
        "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d")
DOG = ("https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/6ddff4824372906469a7fae2dc3206c7aa4bbaee"
       "/assets/dog.jpg", "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866")


def _fetch(url, sha256, folder):
    path = Path(folder) / url.rsplit("/", 1)[1]
    if not path.exists():
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
        except OSError as e:
            pytest.skip(f"cannot download {url}: {e}")
        assert hashlib.sha256(data).hexdigest() == sha256, f"{url} is not the pinned file"
        path.write_bytes(data)
    return path


@pytest.fixture(scope="module")
def nano(tmp_path_factory):
    folder = tmp_path_factory.mktemp("yolox")
    return gray_model(_fetch(*NANO, folder), folder / "nano_gray.onnx"), _fetch(*DOG, folder)


class Scripted:
    """Stands in for a Detector: finds(camera, step) gives each look's finds as
    [(box, score, label index)]. A frame carries its camera in pixel (0, 0) and its
    step in pixel (0, 1)."""
    names = ("person", "car")

    def __init__(self, finds):
        self.finds, self.looks = finds, []

    def __call__(self, frames):
        out = []
        self.looks.append([int(f[0, 0]) for f in frames])
        for f in frames:
            found = self.finds(int(f[0, 0]), int(f[0, 1]))
            out.append((np.array([b for b, _, _ in found], np.float32).reshape(-1, 4),
                        np.array([s for _, s, _ in found], np.float32), np.array([k for _, _, k in found], int)))
        return out


def _frame(cam, step, size=(120, 200)):
    f = np.zeros(size, np.uint8)
    f[0, 0], f[0, 1] = cam, step
    return f


def _walk(step, x=20, speed=5):
    return (x + speed * step, 30, x + 30 + speed * step, 100)


def _run(tracker, steps, cams=(0,), size=(120, 200)):
    return [tracker.step({c: _frame(c, k, size) for c in cams}) for k in range(steps)]


def test_an_object_keeps_its_id_and_has_it_back_after_a_short_absence():
    hidden = range(5, 9)
    det = Scripted(lambda c, k: [] if k in hidden else [(_walk(k), 0.9, 0)])
    out = _run(BoxTracker(det, look_every=1, hold=10), 14)
    assert [sorted(o[0]) for o in out] == [[1]] * 5 + [[]] * 4 + [[1]] * 5
    assert out[13][0][1].label == "person" and abs(out[13][0][1].box[0] - _walk(13)[0]) < 3


def test_an_object_gone_longer_than_hold_comes_back_under_a_new_id():
    det = Scripted(lambda c, k: [] if 5 <= k <= 10 else [(_walk(k), 0.9, 0)])
    out = _run(BoxTracker(det, look_every=1, hold=3), 13)
    assert sorted(out[4][0]) == [1] and sorted(out[12][0]) == [2]


def test_boxes_move_between_looks():
    det = Scripted(lambda c, k: [(_walk(k), 0.9, 0)])
    out = _run(BoxTracker(det, look_every=3), 16)
    lefts = [o[0][1].box[0] for o in out[9:16]]
    assert all(b > a for a, b in zip(lefts, lefts[1:]))  # carried forward on the frames no look sees
    assert abs(lefts[-1] - _walk(15)[0]) < 6


def test_a_new_object_needs_a_second_look_except_on_a_cameras_first():
    car = (120, 40, 170, 80)
    det = Scripted(lambda c, k: [(_walk(k), 0.9, 0)] + ([(car, 0.9, 1)] if k in (3, 6, 7) else []))
    out = _run(BoxTracker(det, look_every=1), 9)
    assert sorted(out[0][0]) == [1]                         # the first look's finds are shown at once
    assert sorted(out[3][0]) == [1] and sorted(out[4][0]) == [1]  # seen once, then not: a false find
    assert sorted(out[6][0]) == [1] and sorted(out[7][0]) == [1, 3] and out[7][0][3].label == "car"


def test_weak_finds_only_confirm_what_is_followed():
    det = Scripted(lambda c, k: [(_walk(k), 0.9 if k < 3 else 0.3, 0), ((150, 10, 190, 60), 0.3, 1)])
    out = _run(BoxTracker(det, look_every=1, hold=0), 8)
    assert [sorted(o[0]) for o in out] == [[1]] * 8  # the person stays seen; the weak car never starts


def test_looks_are_spread_across_cameras():
    det = Scripted(lambda c, k: [])
    _run(BoxTracker(det, look_every=3), 12, cams=range(6))
    assert len(det.looks[0]) == 6                    # every camera's first frame
    assert {len(l) for l in det.looks[1:]} == {2}     # then 6 cameras / 3 frames a step
    assert all(sum(c in l for l in det.looks[k:k + 3]) == 1 for c in range(6) for k in (3, 6))


def test_each_camera_can_have_its_own_look_every():
    det = Scripted(lambda c, k: [])
    _run(BoxTracker(det, look_every={1: 1}), 9, cams=(0, 1))
    assert all(1 in l for l in det.looks) and sum(0 in l for l in det.looks) == 3


def test_a_camera_that_changes_resolution_starts_afresh():
    det = Scripted(lambda c, k: [(_walk(k), 0.9, 0)])
    tracker = BoxTracker(det, look_every=1)
    _run(tracker, 3)
    assert sorted(tracker.step({0: _frame(0, 3, (130, 200))})[0]) == [1]
    assert tracker.cameras[0].next_id == 2 and tracker.cameras[0].size == (130, 200)


def test_bad_settings_are_rejected():
    for kw in ({"look_every": 0}, {"look_every": {"a": 0}}, {"hold": -1}):
        with pytest.raises(ValueError):
            BoxTracker(Scripted(lambda c, k: []), **kw)


def _focus_model(path):
    """A small model shaped as YOLOX starts: Focus (4 interleaved patches stacked), a
    convolution over their 12 channels, and a reshape that names batch 1."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    c = lambda name, v: numpy_helper.from_array(np.array(v, np.int64), name)
    inits = [c("z", [0]), c("o", [1]), c("big", [1 << 30]), c("two", [2]), c("ax2", [2]), c("ax3", [3]),
             numpy_helper.from_array(rng.normal(size=(4, 12, 3, 3)).astype(np.float32), "w"),
             numpy_helper.from_array(rng.normal(size=4).astype(np.float32), "b"), c("shape", [1, 4, -1])]
    nodes = []
    for k, (dy, dx) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        nodes += [helper.make_node("Slice", ["x", "z" if dy == 0 else "o", "big", "ax2", "two"], [f"r{k}"]),
                  helper.make_node("Slice", [f"r{k}", "z" if dx == 0 else "o", "big", "ax3", "two"], [f"p{k}"])]
    nodes += [helper.make_node("Concat", [f"p{k}" for k in range(4)], ["cat"], axis=1),
              helper.make_node("Conv", ["cat", "w", "b"], ["y"], pads=[1, 1, 1, 1]),
              helper.make_node("Reshape", ["y", "shape"], ["out"])]
    g = helper.make_graph(nodes, "focus", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
                          [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 64])], inits)
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 11)]), str(path))
    return path


def test_the_one_channel_model_computes_what_the_three_channel_one_does(tmp_path):
    pytest.importorskip("onnx")
    three = _focus_model(tmp_path / "three.onnx")
    one = gray_model(three, tmp_path / "one.onnx")
    core = ov.Core()
    f32 = {"INFERENCE_PRECISION_HINT": "f32"}
    gray = np.random.default_rng(1).integers(0, 256, (3, 1, 16, 16)).astype(np.float32)
    want = [core.compile_model(str(three), "CPU", f32)({0: np.repeat(g[None], 3, 1)})[0][0] for g in gray]
    got = core.compile_model(str(one), "CPU", f32)({0: gray})[0]  # a batch of 3: the batch is no longer 1
    assert got.shape == (3, 4, 64)
    np.testing.assert_allclose(got, np.stack(want), rtol=1e-5, atol=1e-3)


def test_yolox_nano_finds_on_gray_frames(nano):
    import cv2
    model, dog = nano
    det = Detector(model, threads=1, precision="f32", classes=("dog", "bicycle", "car"))
    bgr = cv2.imread(str(dog))
    gray, rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    (boxes, scores, labels), (_, rgb_scores, _) = det([gray, rgb])
    found = {det.names[k] for k, s in zip(labels, scores) if s > 0.5}
    assert found == {"dog", "bicycle", "car"}
    np.testing.assert_allclose(scores, rgb_scores, atol=0.02)  # RGB is converted to the same gray
    dog_box = boxes[[det.names[k] == "dog" for k in labels]][0]
    assert np.allclose(dog_box, [131, 209, 338, 547], atol=25)  # where the dog is in the picture


def test_yolox_nano_follows_a_panning_camera(nano):
    import cv2
    model, dog = nano
    pic = cv2.imread(str(dog), cv2.IMREAD_GRAYSCALE)
    frames = []
    for k in range(24):
        f = np.full((720, 1280), 120, np.uint8)
        f[60:60 + pic.shape[0], 40 + 6 * k:40 + 6 * k + pic.shape[1]] = pic
        frames.append(f)
    tracker = BoxTracker(Detector(model, threads=1, classes=("dog", "bicycle", "car")), look_every=3)
    out = [tracker.step({"cam": f})["cam"] for f in frames]
    labels = {o.label for o in out[-1].values()}
    assert labels == {"dog", "bicycle", "car"} and set(out[-1]) == {1, 2, 3}  # the same three ids throughout


def test_importing_boxes_does_not_load_torch():
    code = "import sys, tamserve.boxes, tamserve; tamserve.BoxTracker; assert 'torch' not in sys.modules"
    src = str(Path(__file__).parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env={"PYTHONPATH": src, "PATH": ""})
