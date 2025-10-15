import torch
import numpy as np
import pytest
from airbus_ship_detection import metrics

def test_compute_iou_basic():
    pred = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    true = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    iou = metrics.compute_iou(pred, true)
    expected = 1 / (4 - 1 + metrics.SMOOTHING)
    assert np.isclose(iou.item(), expected, atol=1e-6)

def test_compute_fscore_perfect():
    tp, fn, fp = 10, 0, 0
    f2 = metrics.compute_fscore(tp, fn, fp, beta=2)
    assert np.isclose(f2, 1.0, atol=1e-6)

def test_compute_fscore_no_objects():
    tp, fn, fp = 0, 0, 0
    f2 = metrics.compute_fscore(tp, fn, fp, beta=2)
    assert f2 == 1.0

def test_compute_confusion_matrix_counts_all_zeros():
    pred = torch.zeros(16, 256)
    true = torch.zeros(16, 256)
    tp, fn, fp = metrics.compute_confusion_matrix_counts(pred, true, batch_size=16, threshold=0.5)
    assert tp == 0
    assert fn == 0
    assert fp == 0

def test_compute_confusion_matrix_counts_simple():
    pred = torch.zeros(16, 256)
    true = torch.zeros(16, 256)
    pred[0, 0] = 1.0
    tp, fn, fp = metrics.compute_confusion_matrix_counts(pred, true, batch_size=16, threshold=0.5)
    assert tp == 0
    assert fn == 0
    assert fp == 1

def test_compute_mean_fscore():
    pred = torch.zeros(16, 256)
    true = torch.zeros(16, 256)
    pred[0, 0] = 1.0
    score = metrics.compute_mean_fscore(pred, true, iou_thresholds=[0.5], batch_size=16, threshold=0.5, beta=2)
    assert 0 <= score <= 1

def test_compute_dice_jaccard_all_zeros():
    pred = torch.zeros(16, 256)
    true = torch.zeros(16, 256)
    dice, jaccard = metrics.compute_dice_jaccard(pred, true, batch_size=16, threshold=0.5)
    assert torch.allclose(dice, torch.ones_like(dice))
    assert torch.allclose(jaccard, torch.ones_like(jaccard))

def test_compute_dice_jaccard_perfect_match():
    pred = torch.ones(16, 256)
    true = torch.ones(16, 256)
    dice, jaccard = metrics.compute_dice_jaccard(pred, true, batch_size=16, threshold=0.5)
    assert torch.allclose(dice, torch.ones_like(dice))
    assert torch.allclose(jaccard, torch.ones_like(jaccard))

def test_metrics_class_collect_and_get():
    m = metrics.Metrics(batch_size=2, threshold=0.5)
    pred = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    true = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    m.collect(pred, true)
    meanf2, dice, jaccard = m.get()
    assert 0 <= meanf2 <= 1
    assert 0 <= dice <= 1
    assert 0 <= jaccard <= 1

def test_metrics_class_multiple_collects():
    m = metrics.Metrics(batch_size=2, threshold=0.5)
    pred1 = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    true1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    pred2 = torch.tensor([[10.0, 10.0], [0.0, 0.0]])
    true2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    m.collect(pred1, true1)
    m.collect(pred2, true2)
    meanf2, dice, jaccard = m.get()
    assert 0 <= meanf2 <= 1
    assert 0 <= dice <= 1
    assert 0 <= jaccard <= 1