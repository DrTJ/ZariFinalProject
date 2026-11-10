"""Unit tests for the cascade logic and metrics. No neural networks needed:
fake detectors return scripted detections."""
import numpy as np
import pytest

from cascade_mot.cascade import Cascade, CascadeConfig, FullOnly, LiteOnly
from cascade_mot.detectors import Detections
from cascade_mot.metrics import agreement, iou_matrix, match_count


def dets(*scores, label=1):
    n = len(scores)
    boxes = np.array([[10 * i, 0, 10 * i + 8, 8] for i in range(n)], np.float32).reshape(-1, 4)
    return Detections(boxes, np.array(scores, np.float32), np.full(n, label, np.int64))


class Scripted:
    def __init__(self, seq):
        self.seq, self.i, self.calls = seq, 0, 0

    def __call__(self, frame):
        d = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1; self.calls += 1
        return d


def run(lite_seq, cfg, full_det=None):
    lite = Scripted(lite_seq)
    full = Scripted([full_det or dets(0.9)] * len(lite_seq))
    c = Cascade(lite, full, cfg)
    infos = [c(None)[1] for _ in lite_seq]
    return [i.reason for i in infos], full.calls


def test_first_frame_uses_full():
    reasons, calls = run([dets(0.9)], CascadeConfig())
    assert reasons == ["first"] and calls == 1


def test_stable_confident_scene_stays_on_lite():
    reasons, calls = run([dets(0.9, 0.8)] * 10, CascadeConfig(refresh_interval=0))
    assert reasons == ["first"] + ["lite"] * 9 and calls == 1


def test_uncertain_score_triggers_full():
    seq = [dets(0.9), dets(0.9), dets(0.9, 0.45), dets(0.9)]
    reasons, _ = run(seq, CascadeConfig(refresh_interval=0))
    assert reasons == ["first", "lite", "uncertain", "lite"]


def test_scores_below_low_are_ignored():
    reasons, _ = run([dets(0.9), dets(0.9, 0.1)], CascadeConfig(refresh_interval=0))
    assert reasons == ["first", "lite"]


def test_count_change_triggers_full_and_tolerance():
    seq = [dets(0.9), dets(0.9, 0.9), dets(0.9, 0.9)]
    assert run(seq, CascadeConfig(refresh_interval=0))[0] == ["first", "count_change", "lite"]
    assert run(seq, CascadeConfig(refresh_interval=0, count_tolerance=1))[0] == ["first", "lite", "lite"]


def test_keyframe_refresh():
    reasons, calls = run([dets(0.9)] * 7, CascadeConfig(refresh_interval=3))
    assert reasons == ["first", "lite", "lite", "lite", "keyframe", "lite", "lite"]
    assert calls == 2


def test_output_comes_from_the_model_that_was_used():
    lite = Scripted([dets(0.9), dets(0.9)])
    full = Scripted([dets(0.95, 0.94, 0.93)])
    c = Cascade(lite, full, CascadeConfig(refresh_interval=0))
    assert len(c(None)[0]) == 3      # first frame -> Full's 3 boxes
    assert len(c(None)[0]) == 1      # then Lite's 1 box


def test_conf_threshold_applied():
    d = dets(0.9, 0.55, 0.4)
    assert len(FullOnly(Scripted([d]), conf=0.5)(None)[0]) == 2
    assert len(LiteOnly(Scripted([d]), conf=0.6)(None)[0]) == 1


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        Cascade(Scripted([dets()]), Scripted([dets()]), CascadeConfig(low=0.7, high=0.6))


def test_iou_and_matching():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3]], rtol=1e-5)
    ref = Detections(a, np.array([0.9], np.float32), np.array([1]))
    other_class = Detections(a, np.array([0.9], np.float32), np.array([3]))
    assert match_count(ref, ref) == 1
    assert match_count(ref, other_class) == 0


def test_mode_agrees_with_itself_even_with_degenerate_boxes():
    boxes = np.array([[5, 5, 5, 9], [0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    d = Detections(boxes, np.array([0.9, 0.8, 0.7], np.float32), np.array([1, 1, 1]))
    assert agreement([d], [d])["recall_vs_full"] == 1.0


def test_agreement_perfect_and_partial():
    ref = [dets(0.9, 0.9), dets(0.9)]
    assert agreement(ref, ref)["recall_vs_full"] == 1.0
    partial = [dets(0.9), Detections()]
    m = agreement(ref, partial)
    assert m["recall_vs_full"] == pytest.approx(1 / 3)
    assert m["precision_vs_full"] == 1.0


def test_mot_evaluator_perfect_tracking(tmp_path):
    pytest.importorskip("motmetrics")
    from cascade_mot.metrics import MOTEvaluator
    from cascade_mot.tracking import TrackOut
    gt = tmp_path / "gt.txt"
    gt.write_text("1,1,0,0,10,10,1,1,1\n2,1,1,0,10,10,1,1,1\n2,2,50,50,10,10,1,1,1\n")
    ev = MOTEvaluator(str(gt))
    ev.update(0, [TrackOut(7, (0, 0, 10, 10), 1)])
    ev.update(1, [TrackOut(7, (1, 0, 11, 10), 1), TrackOut(8, (50, 50, 60, 60), 1)])
    s = ev.summary()
    assert s["mota"] == pytest.approx(1.0) and s["num_switches"] == 0 and s["idf1"] == pytest.approx(1.0)
