## Methods for preprocessing and prepping data for a tf pipeline.

from scipy import spatial
from skimage import measure
from scipy import ndimage
import numpy as np
from sklearn import metrics

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
    iou_max = np.amax(ious, axis=0)
    ## fp + tn = number of predicted features
    fp = ious.shape[1] - tp
    fn = ious.shape[0] - tp
    return np.array([tp, fp, fn]), iou_max

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
  ious = []
  for i in range(N):
    mask_pred = np.reshape(masks_pred[i] > 0.5, (size, size))
    mask = np.reshape(masks[i], (size, size))
    img_counts, img_ious = feature_counts(mask, mask_pred, threshold=threshold) 
    counts = counts + [img_counts]
    ious = ious + [img_ious]
  counts = np.stack(counts, axis=0)
  ious = np.concatenate(ious)
  iou_sd = np.std(ious)
  return np.append(_feature_metrics(counts), [threshold, iou_sd])

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
  sd = np.std(np.stack(metrics, axis=0), axis=0)
  return np.append(metrics, [threshold, sd])

def pixel_ap(masks, masks_pred):
    return metrics.average_precision_score(masks.reshape((-1)), masks_pred.reshape((-1)))
    
def all_pixel_metrics(masks, masks_pred, thresholds):
    return np.stack([pixel_metrics(masks, masks_pred, t) for t in thresholds], axis = 0)


def intersection(bb1, bb2):
    '''
    bb1 and bb2: 2D np array of bounding boxes in format (top, left, bottom, right). NOTE: assuming
    non-zero areas and non-empty arrays. Respective sizes N x 4 and M x 4.
    '''
    screen_coords = bb1[0,0] < bb1[0,2] ## If top is lesser than bottom coordinate, we are using
    ## screen coordinates.
    y_max_fn = np.maximum if screen_coords else np.minimum
    y_min_fn = np.minimum if screen_coords else np.maximum

    ## A projection (selecting a column).
    def p(bb, i):
        return bb[:, i].reshape((-1, 1))

    t = np.transpose

    ## Each of these is NxN
    top = y_max_fn(p(bb1, 0), t(p(bb2, 0)))
    bottom = y_min_fn(p(bb1, 2), t(p(bb2, 2)))
    left = np.maximum(p(bb1, 1), t(p(bb2, 1)))
    right = np.minimum(p(bb1, 3), t(p(bb2, 3)))

    x_overlaps = np.maximum(right - left, 0)
    y_overlaps = y_max_fn(bottom - top, 0)

    intersection = (x_overlaps * y_overlaps)

    return intersection

def area(bb):
    ## A projection (selecting a column).
    def p(i):
        return bb[:, i].reshape((-1, 1))
    return np.absolute((p(0) - p(2)) * (p(1) - p(3)))


def bbox_occlusion(bb, threshold = 0.2):
    '''
    bb: A 2D np array of bounding boxes in format (top, left, bottom, right). NOTE: assuming
    non-zero areas.
    '''

    ## A note on numpy broadcasting: Numpy aligns the trailing dimensions of arrays. All aligned
    ## dimensions must be equal or one of them should be 1.  The leading dimensions of the lower
    ## dimensional array are assumed to be 1. Any dimension equal to 1 is "stretched" (array is
    ## repeated along that dimension) till the dimension is equal to the corresponding dimension of
    ## the other array. This is just a looping construct to ensure that looping happens in C and not
    ## in python.
    ## See: https://numpy.org/doc/stable/user/basics.broadcasting.html

    inter = intersection(bb, bb)
    ## Ignore self intersections
    np.fill_diagonal(inter, 0)

    ## This union is obviously wrong. We can fix it by using masks in a later iteration.
    ## As we are working in discrete image coordinates here, it is feasible. On the other hand,
    ## current definition penalizes multiple occlusions, so maybe this definition is okay too.
    total_occlusion = np.sum(inter, axis = 1).reshape((-1, 1))

    return total_occlusion/area(bb) > threshold

def test_bbox_occlusion():
    bboxes = np.array([1, 1, 3, 3, 3, 1, 5, 3, 3, 3, 5, 5, 1, 3, 3, 5, 2, 2, 4, 4]
                      ).reshape((-1, 4))

    occ_3 = bbox_occlusion(bboxes, threshold = 0.3)
    occ_2 = bbox_occlusion(bboxes, threshold = 0.2)

    assert np.all(occ_2)
    assert np.all(occ_3 == np.array([False, False, False, False, True]).reshape((-1, 1)))


def iou(bb1, bb2):
    '''
    bb1: N x 4 array of bounding boxes.
    bb2: M x 4 array of bounding boxes.
    '''
    intersections = intersection(bb1, bb2)
    unions = area(bb1) + np.transpose(area(bb2)) - intersections
    return intersections/unions

def detected_bboxes(bboxes, detections, iou_threshold=0.5):
    '''
    Map detections to bboxes and bboxes to detections
    iou_threshold: should be >= 0.5
    '''

    def fix(x, dtype=np.int32):
        return x.reshape((-1, 1)).astype(dtype) if x.size == 0 else x.reshape((-1, 1))
    ious = iou(bboxes, detections[:, range(4)])
    bboxes_to_detections = np.argmax(ious, axis=1)
    detections_to_bboxes = np.argmax(ious, axis=0)
    detected_boxes = np.amax(ious, axis=1) >= iou_threshold
    detection_indices = fix(np.argmax(ious, axis=1)[detected_boxes])
    bboxes_without_detections = fix(np.arange(bboxes.shape[0])[
        np.logical_not(detected_boxes)])
    detections_without_bboxes = fix(
        np.setdiff1d(np.arange(detections.shape[0]), detection_indices))
    dt = detections[:,4].dtype
    detected_confidence = fix(detections[detection_indices, 4], dtype=dt)
    bboxes_without_detections_confidence = fix(
        np.repeat(0.0, bboxes_without_detections.shape[0]), dtype=dt) 
    detections_without_bboxes_confidence = fix(detections[detections_without_bboxes, 4], dtype=dt)
    confidences = np.concatenate(
        (detected_confidence,
         bboxes_without_detections_confidence,
         detections_without_bboxes_confidence), axis=0)
    x = bbox_occlusion(bboxes)
    occluded = np.concatenate(
        (x[np.nonzero(detected_boxes)],
         x[np.nonzero(np.logical_not(detected_boxes))],
         np.repeat(False, confidences.shape[0] - x.shape[0]).reshape(-1, 1)), axis=0)
    labels = np.concatenate(
        (np.repeat(1, bboxes.shape[0]).reshape((-1, 1)),
         np.repeat(0, confidences.shape[0] - bboxes.shape[0]).reshape((-1,1))), axis=0)
    return np.concatenate((labels, confidences, occluded), axis=1)
    
def per_image_predictions(bboxes, detections, iou_threshold = 0.5):
    '''
    bboxes: Ground truth bboxes N x 4 array of (t, l, b, r) values.
    detections: M x 5 array of (t, l, b, r, c) values.
    iou_threshold: Assume a positive detection above this value.
    '''
    ious = iou(bboxes, detections[:,range(4)])
    return detected_bboxes(bboxes, detections, iou_threshold=iou_threshold)

def test_per_image_predictions():
    bboxes = np.array([1, 1, 2, 2, 3, 1, 5, 3, 3, 3, 5, 5, 1, 3, 3, 5, 2, 2, 4, 4]
                      ).reshape((-1, 4))
    detections = np.array(
        [3, 1, 5, 3, 0.51, 1, 1, 2.1, 2.1, 0.5, 3, 3, 5, 5, 0.49, 10, 30, 30, 50, 0.52,
         100, 300, 300, 500, 0.48, 1000, 3000, 3000, 5000, 0.47]
    ).reshape((-1, 5))
    pip = per_image_predictions(bboxes, detections) 
    assert np.all((pip == np.array([
        [1., 0.5, 0.],
        [1., 0.51, 1.],
        [1., 0.49, 1.],
        [1., 0.0, 1.],
        [1., 0.0, 1.],
        [0., 0.52, 0.],
        [0., 0.48, 0.],
        [0., 0.47, 0.]])))

def _ap(precision, recall):
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])
    
def ap(image_gt_and_predictions, iou_threshold=0.5):
    '''
    image_gt_and_predictions: A list of tuples. Each tuple consists of the ground truth bboxes and
    detections.
    '''
    all_predictions = np.concatenate(list(map(lambda x: per_image_predictions(x[0], x[1], iou_threshold),
                                              image_gt_and_predictions)))
    occluded = all_predictions[all_predictions[:,2] == 1.0, :]
    non_occluded = all_predictions[all_predictions[:,2] == 0.0, :]
    prt = metrics.precision_recall_curve(all_predictions[:, 0], all_predictions[:, 1])
    prt_occ = metrics.precision_recall_curve(occluded[:, 0], occluded[:, 1])
    occ_threshold_idxs = np.searchsorted(prt[2], prt_occ[2])
    prt_no_occ = metrics.precision_recall_curve(non_occluded[:, 0], non_occluded[:, 1])
    no_occ_threshold_idxs = np.searchsorted(prt[2], prt_no_occ[2])
    occ_precision = np.hstack((prt[0][occ_threshold_idxs], 0))
    occluded = all_predictions[all_predictions[:,2] == 1.0, :]
    no_occ_precision = np.hstack((prt[0][no_occ_threshold_idxs], 0))
    return (_ap(prt[0], prt[1]),
            _ap(occ_precision, prt_occ[1]),
            _ap(no_occ_precision, prt_no_occ[1])) 
    
def test_ap():
    bboxes1 = np.array([1, 1, 3, 3,
                        3, 1, 5, 3,
                        3, 3, 5, 5,
                        1, 3, 3, 5,
                        2, 2, 4, 4,
                        101, 301, 301, 501,
                        10001, 10002, 10003, 10004]
                       ).reshape((-1, 4))
    detections1 = np.array(
        [1, 1, 3, 3, 0.6,
         3, 1, 5, 3, 0.61,
         3, 3, 5, 5, 0.59,
         10, 30, 30, 50, 0.42,
         100, 300, 300, 500, 0.38,
         1000, 3000, 3000, 5000, 0.37,
         10001, 10002, 10003, 10004, 0.57]
    ).reshape((-1, 5))
    bboxes2 = np.array([1, 1, 3, 3,
                        3, 1, 5, 3,
                        3, 3, 5, 5,
                        1, 3, 3, 5,
                        2, 2, 4, 4]
                       ).reshape((-1, 4))
    detections2 = np.array(
        [1, 1, 3, 3, 0.5,
         3, 1, 5, 3, 0.51,
         3, 3, 5, 5, 0.49,
         10, 30, 30, 50, 0.52,
         100, 300, 300, 500, 0.48,
         1000, 3000, 3000, 5000, 0.47,
         1001, 3000, 3000, 5000, 0.56]
    ).reshape((-1, 5))
    print(bboxes1.shape)
    print(bboxes2.shape)
    print(detections1.shape)
    print(detections2.shape)
    image_gt_and_predictions = [(bboxes1, detections1), (bboxes2, detections2)]
    return ap(image_gt_and_predictions)


def dataset_ap(gt_files, predictions_files, gt_file_reader, predictions_file_reader):
    ''' 
    gt_files: a list of files, each with ground truth bounding boxes for one images.
    predictions_files: a list of files, each with predicted bounding boxes and objectness
    score. These predictions should correspond 1-1 with the ground truth files.
    gt_file_reader: a function that takes path to a single ground truth file and returns 
    a Nx4 numpy array of bounding boxes.
    predictions_file_reader: a function that takes path to a single prediction file and returns
    a Mx5 numpy array of bounding boxes and objectness scores.
    '''
    pass

    
