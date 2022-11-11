### Methods for preprocessing and prepping data for a tf pipeline.

from scipy import spatial
from skimage import measure
from scipy import ndimage
import numpy as np

def as_np(x):
    return x.numpy() if type(x) != np.ndarray else x

def mask_sizes(mask):
    labels = measure.label(as_np(mask))
    return np.unique(labels, return_counts=True)

def filter_tiny_masks(mask, threshold = 50):
    labels = measure.label(as_np(mask))
    sizes = np.unique(labels, return_counts=True)
    tiny = np.asarray(sizes[1] <= threshold).nonzero()
    for t in tiny[0]:
        labels[labels == t] = 0
    labels[labels > 0] = 1
    return labels

def feature_iou(mask_true, mask_pred):
    mask_pred = as_np(mask_pred)
    mask_true = as_np(mask_true)
    mask_pred = filter_tiny_masks(mask_pred)
    pred_labels = measure.label(mask_pred) - 1
    true_labels = measure.label(mask_true) - 1
    pred_sizes = np.unique(pred_labels, return_counts=True)[1][1:]
    true_sizes = np.unique(true_labels, return_counts=True)[1][1:]
    intersections = np.zeros((len(true_sizes), len(pred_sizes)))
    for i in range(len(true_sizes)):
        for j in range(len(pred_sizes)):
            intersections[i , j] = np.sum((pred_labels == j) & (true_labels == i))
    unions = np.zeros((len(true_sizes), len(pred_sizes)))
    for i in range(len(true_sizes)):
        for j in range(len(pred_sizes)):
            unions[i , j] = np.sum((pred_labels == j) | (true_labels == i))
    return intersections/unions

def feature_counts(mask_true, mask_pred, threshold = 0.5):
    ious = feature_iou(mask_true, mask_pred)
    tp = np.sum(np.any(ious > threshold, axis = 0))
    ## fp + tn = number of predicted features
    fp = ious.shape[1] - tp
    fn = ious.shape[0] - tp
    return np.array([tp, fp, fn])

def _feature_metrics(counts):
    n = np.sum(counts, axis = 0)
    tp = n[0]
    fp = n[1]
    fn = n[2]
    precision = tp/(tp + fn)
    recall = tp/(tp + fp)
    intersection = tp
    union = tp + fn + fn
    iou = intersection/union
    accuracy = tp/(tp + fp + fn)
    return np.array([precision, recall, iou, accuracy])

def feature_metrics(masks, masks_pred, threshold, size = 512):
  N = masks.shape[0]
  counts = []
  for i in range(N):
    mask_pred = np.reshape(masks_pred[i] > 0.5, (size, size))
    mask = np.reshape(masks[i], (size, size))
    counts = counts + [feature_counts(mask, mask_pred, threshold = threshold)]
  counts = np.stack(counts, axis=0)
  return np.append(_feature_metrics(counts), [threshold])

def all_feature_metrics(masks, masks_pred, thresholds, size = 512):
    return np.stack([feature_metrics(masks, masks_pred, t) for t in thresholds])


def _pixel_metrics(mask, mask_pred, size = 512):
    mask = np.reshape(as_np(mask), (size, size))
    mask_pred = np.reshape(as_np(mask_pred), (size, size))
    intersection = np.sum((mask == 1) &  (mask_pred == 1))
    union = np.sum(mask) + np.sum(mask_pred) - intersection
    iou = intersection/union
    tp = intersection
    fp = np.sum((mask_pred == 1) & (mask == 0))
    tn = np.sum((mask_pred == 0) & (mask == 0))
    fn = np.sum((mask_pred == 0) & (mask == 1))
    recall = tp/(tp + fn)
    precision = tp/(tp + fp)
    absolute_area_err = np.abs(1- np.sum(mask_pred)/np.sum(mask))
    area_err =(1- np.sum(mask_pred)/np.sum(mask))
    return (precision, recall, iou, np.sum(mask == mask_pred)/(size*size), absolute_area_err,
            area_err)

def pixel_metrics(masks, masks_pred, threshold):
  N = masks.shape[0]
  metrics = []
  for i in range(N):
    mask_pred = masks_pred[i] > threshold
    metrics = metrics + [_pixel_metrics(masks[i], mask_pred)]
  metrics = np.mean(np.stack(metrics, axis=0), axis = 0)
  return np.append(metrics, [threshold])

def all_pixel_metrics(masks, masks_pred, thresholds):
    return np.stack([pixel_metrics(masks, masks_pred, t) for t in thresholds], axis = 0)
