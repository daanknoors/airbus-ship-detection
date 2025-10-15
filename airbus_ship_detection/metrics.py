import torch
import numpy as np
SMOOTHING = 1e-8 # to avoid division by zero

def compute_iou(pred, true):
    # intersection over union
    pred = (pred > 0.5).float()
    intersection = (pred * true).sum()
    return intersection / ((pred + true).sum() - intersection + SMOOTHING)

def compute_fscore(tp, fn, fp, beta=2):
    # no objects in both pred and true
    if (tp + fn + fp) == 0:
        return float(1)
    numerator = (1 + beta**2) * tp
    denominator = (1 + beta**2) * tp + (beta**2 * fn) + fp + SMOOTHING
    return numerator / denominator

def compute_confusion_matrix_counts(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)

    # if no objects in true, then only fp is relevant
    if true.sum() == 0:
        fp = (pred > threshold).float().sum()
        return 0, 0, fp
    
    # compute tp, fn, fp with IoU above threshold
    pred = (pred > threshold).float()
    true = (true > threshold).float()
    tp = ((pred + true) == 2).float().sum()
    fn = ((pred + true) == 1).float().sum() * (true == 1).float().sum() / true.numel()
    fp = ((pred + true) == 1).float().sum() * (pred == 1).float().sum() / pred.numel()
    return tp, fn, fp

def compute_mean_fscore(pred, true, iou_thresholds, batch_size=16, threshold=0.5, beta=2):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)
    
    f2score = compute_fscore(*compute_confusion_matrix_counts(pred, true, batch_size=batch_size, threshold=threshold), beta=beta)
    return f2score

def compute_dice_jaccard(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)
    
    pred = (pred > threshold).float()
    true = (true > threshold).float()
    
    pred_sum = pred.sum(-1)
    true_sum = true.sum(-1)
    
    neg_index = torch.nonzero(true_sum == 0)
    pos_index = torch.nonzero(true_sum >= 1)
    
    dice_neg = (pred_sum == 0).float()
    dice_pos = 2 * ((pred * true).sum(-1)) / ((pred + true).sum(-1))
    
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    
    dice = torch.cat([dice_pos, dice_neg])
    jaccard = dice / (2 - dice)
    
    return dice, jaccard

class Metrics:
    def __init__(self, batch_size=16, threshold=0.5):
        self.threshold = threshold
        self.batchsize = batch_size

        self.mean_f2score = []
        self.dice = []
        self.jaccard = []

    def collect(self, pred, true):
        pred = torch.sigmoid(pred)
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        mean_f2score = compute_mean_fscore(pred, true, iou_thresholds, batch_size=self.batchsize, threshold=self.threshold, beta=2)
        dice, jaccard = compute_dice_jaccard(pred, true, batch_size=self.batchsize, threshold=self.threshold)

        self.mean_f2score.append(mean_f2score)
        self.dice.extend(dice)
        self.jaccard.extend(jaccard)

    def get(self):
        mean_f2score = np.nanmean(self.mean_f2score)
        dice = np.nanmean(self.dice)
        jaccard = np.nanmean(self.jaccard)
        return mean_f2score, dice, jaccard
