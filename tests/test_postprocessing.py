import numpy as np

from vision.config import get_settings
from vision.models.base import RawDetection
from vision.postprocessing.nms import (box_iou, classwise_nms, filter_by_threshold, merge_fragments,
                                       suppress_contained_duplicates)
from vision.taxonomy import load_taxonomy


def R(box, s, c="staircase", p="stairs"):
    return RawDetection(np.array(box, float), s, c, p, "m")


def test_iou():
    assert box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == 1.0
    assert abs(box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] - 1 / 3) < 1e-9
    assert box_iou([[0, 0, 1, 1]], [[5, 5, 6, 6]])[0, 0] == 0


def test_synonym_prompts_collapse_but_classes_kept():
    dets = [R([0, 0, 100, 100], 0.9, p="stairs"), R([2, 2, 101, 99], 0.7, p="staircase"),
            R([0, 0, 100, 100], 0.6, c="handrail", p="handrail")]
    out = classwise_nms(dets, 0.5)
    assert len(out) == 2
    st = [d for d in out if d.class_name == "staircase"][0]
    assert st.score == 0.9 and st.prompt == "stairs"


def test_threshold():
    dets = [R([0, 0, 10, 10], 0.3), R([0, 0, 10, 10], 0.6)]
    assert len(filter_by_threshold(dets, lambda d: 0.5)) == 1


def test_merge_fragments_per_step_boxes():
    steps = [R([100, 100 + 20 * i, 300, 118 + 20 * i], 0.2 + 0.01 * i) for i in range(8)]
    other = R([500, 0, 520, 100], 0.5, c="handrail", p="handrail")
    out = merge_fragments(steps + [other], {"staircase"})
    st = [d for d in out if d.class_name == "staircase"]
    assert len(st) == 1 and st[0].xyxy.tolist() == [100, 100, 300, 258] and abs(st[0].score - 0.27) < 1e-9
    assert any(d.class_name == "handrail" for d in out)


def test_merge_fragments_keeps_separate_regions():
    a, b = R([0, 0, 100, 100], 0.5), R([400, 0, 500, 100], 0.5)
    assert len(merge_fragments([a, b], {"staircase"})) == 2


def test_contained_duplicate_suppressed_but_nested_small_kept():
    big = R([0, 0, 100, 200], 0.9)
    dup = R([5, 5, 95, 190], 0.5)         # same object, IoU<0.5? no: inside and similar size
    small = R([10, 10, 30, 30], 0.4)      # genuinely nested smaller object
    out = suppress_contained_duplicates([big, dup, small])
    ids = {id(d) for d in out}
    assert id(big) in ids and id(small) in ids and id(dup) not in ids


def test_taxonomy_prompts_unique_and_mapped():
    tax = load_taxonomy(str(get_settings().taxonomy_config))
    prompts = tax.all_prompts()
    assert len(prompts) == len(set(p.lower() for p in prompts))
    for c in tax.classes.values():
        assert c.prompts or c.coco_labels, c.name
        for p in c.prompts:
            assert tax.class_for_prompt(p) == c.name
    assert tax.class_for_coco("person") == "person" and tax.class_for_coco("chair") == "obstacle"
