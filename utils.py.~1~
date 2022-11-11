import tensorflow as tf

def tile(vec, n_columns):
  return tf.tile(tf.reshape(vec, (vec.shape[0], 1)), [1, n_columns])

def iou(bboxes1, bboxes2):
  '''
    bboxes1: (N, 4)
    bboxes2: (M, 4)
    output: IOUs for all prediction, gt combinations (N, M)
  '''
  ## top y-coordinates of bboxes1
  y1_t = bboxes1[:,0]
  ## left x-coordinates of bboxes1 (and so on)
  x1_l = bboxes1[:,1]
  y1_b = bboxes1[:,2]
  x1_r = bboxes1[:,3]

  y2_t = bboxes2[:,0]
  x2_l = bboxes2[:,1]
  y2_b = bboxes2[:,2]
  x2_r = bboxes2[:,3]

  areas1 = (y1_b - y1_t) * (x1_r - x1_l)
  areas2 = (y2_b - y2_t) * (x2_r - x2_l)

  overlaps = tf.math.maximum(
      0,
      tf.math.minimum(tile(y1_b, bboxes2.shape[0]), y2_b) -
          tf.math.maximum(tile(y1_t, bboxes2.shape[0]), y2_t))
  overlaps = overlaps * tf.math.maximum(
      0,
      tf.math.minimum(tile(x1_r, bboxes2.shape[0]), x2_r) -
          tf.math.maximum(tile(x1_l, bboxes2.shape[0]), x2_l))

  return overlaps/((tf.reshape(areas1, (areas1.shape[0],1 )) +
      tf.reshape(areas2, (1, areas2.shape[0]))) - overlaps)

def centered(boxes):
  '''
  Convert box from corner (y_min, x_min, y_max, x_max) to centered (y, x, h, w)
  format.
  '''
  ys = (boxes[:,2] + boxes[:,0])/2.
  xs = (boxes[:,3] + boxes[:,1])/2.
  hs = boxes[:,2] - boxes[:,0]
  ws = boxes[:,3] - boxes[:,1]
  return tf.stack([ys, xs, hs, ws], axis=1)

def corner(boxes):
  '''
  Convert box from centered (y, x, h, w) to corner (y_min, x_min, y_max, x_max)
  format.
  '''
  y_min = boxes[:, 0] - boxes[:, 2]/2.
  x_min = boxes[:, 1] - boxes[:, 3]/2.
  y_max = boxes[:, 0] + boxes[:, 2]/2.
  x_max = boxes[:, 1] + boxes[:, 3]/2.

  return tf.stack([y_min, x_min, y_max, x_max], axis=1)

def perturbations(anchors, boxes):
  '''
  anchors: [N, 4] in corner format
  boxes: [N, 4] in corner format
  Returns perturbations for anchors for given reference boxes.
  This is as defined in Section 3.1.2 of https://arxiv.org/pdf/1506.01497.pdf
  '''
  a = centered(anchors)
  b = centered(boxes)
  ty = (b[:,0] - a[:,0])/a[:, 2]
  tx = (b[:,1] - a[:,1])/a[:, 3]
  th = tf.math.log(b[:,2]/a[:, 2])
  tw = tf.math.log(b[:,3]/a[:, 3])
  return tf.stack([ty, tx, th, tw], axis = 1)

def anchor_gt_assignment(anchors, gt_boxes, N):
  '''
  anchors: of shape [num_anchors, 4]
  gt_boxes: of shape [num_gt_boxes, 4]
  outputs:
  positive_anchors: [n <= N, (anchor_index, matched_gt_box_index)]
  deltas: [n <= N, perturbations]
  negative_anchors: [n <= m <= N, anchor_index]
  perturbations, as defined in Function "perturbations", required to match a
  positive anchor to corresponding gt box. Neutral anchors aren't used for
  training. Samples are used to train RPN. Sampling maintains balance between
  negative and positive anchors.
  Question: Is a positive anchor always assigned to a single gt box,
  ensuring that the perturbations for an anchor are well defined?
  Answer: For computing perturbations, an anchor is assigned to one gt box with
  which it has the maximum overlap. When computing loss for regression layer,
  only positive anchors are used.
  For some details on anchor<-> gt_box attribution see Section 3.1.2 of
  https://arxiv.org/pdf/1506.01497.pdf
  '''
  ious = iou(anchors, gt_boxes)
  ## max anchor ious for a given gt box
  best_scores = tf.math.reduce_max(ious, axis=0)
  best_anchors = tf.range(anchors.shape[0]) == tf.cast(tf.argmax(ious, axis=0), dtype=tf.int32)
  anchors_above_threshold = tf.reduce_any(ious >= 0.7, axis=1)
  positive_anchors = tf.logical_or(anchors_above_threshold, best_anchors)
  negative_anchors = tf.logical_and(
      tf.reduce_all(ious < 0.3, axis = 1),
      tf.logical_not(positive_anchors))
  n_positive_anchors = tf.reduce_sum(tf.cast(positive_anchors, dtype=tf.int32))
  n_negative_anchors = tf.reduce_sum(tf.cast(negative_anchors, dtype=tf.int32))
  ## Don't sample more positive anchors than negative anchors
  num_positive_samples = tf.math.reduce_min([
    n_negative_anchors, n_positive_anchors, N
  ])
  ## Don't sample more negative samples than 3 times the positive samples
  num_negative_samples = tf.math.reduce_min([
    3*n_positive_anchors, n_negative_anchors, N
  ])
  positive_anchor_indices = tf.cast(tf.reshape(tf.random.shuffle(tf.where(positive_anchors))[
    0:num_positive_samples
  ], [num_positive_samples]), dtype=tf.int32)
  negative_anchor_indices = tf.cast(tf.reshape(tf.random.shuffle(tf.where(negative_anchors))[
    0:num_negative_samples
  ], [num_negative_samples]), dtype=tf.int32)
  ## Best GT boxes for a given anchor
  best_gt_box_indices = tf.cast(tf.argmax(tf.gather(ious, positive_anchor_indices, axis=0), axis=1), dtype=tf.int32)
  best_gt_boxes = tf.gather(gt_boxes, best_gt_box_indices)
  positive_anchor_coords = tf.gather(anchors, positive_anchor_indices)
  deltas = perturbations(positive_anchor_coords, best_gt_boxes)
  return tf.stack([positive_anchor_indices, best_gt_box_indices], axis=1), deltas, negative_anchor_indices

def anchor_centers(shape, stride):
  '''
  shape: H, W of the image over which anchors are to be generated
  feature_stride: Stride (in pixels) of feature map relative to the image
  output: points (in image pixel coordinates) over which the anchors are to be
  generated
  '''
  ys, xs = tf.meshgrid(tf.range(stride//2, shape[0], stride, dtype=tf.float32),
      tf.range(0, shape[1], stride, dtype=tf.float32))
  ## xs.shape == ys.shape
  return tf.reshape(ys, [-1]), tf.reshape(xs, [-1])


def anchors(shape, sizes, ratios, stride):
  ys, xs = anchor_centers(shape, stride)
  ## Anchors of given size and ratio
  def f(size, ratio):
    r = tf.math.sqrt(ratio)
    ## w*h = size*size; w/h = ratio
    w = tf.ones([xs.shape[0]])*r*size
    h = tf.ones([ys.shape[0]])*size/r
    return corner(tf.stack([ys, xs, h, w], axis=1))
  ## Anchors of given size
  return tf.concat([f(size, ratio) for size in sizes for ratio in ratios], axis=0)

## Clip anchors at image boundaries
def clip_anchors(anchors, min=[0,0], max=[512, 512]):
  idx = tf.where(
      tf.logical_and(
          anchors[:,0] >= min[0],
          tf.logical_and(
              anchors[:,1] >= min[1],
              tf.logical_and(anchors[:,2] <= max[0], anchors[:,3] <= max[1]))))
  return tf.gather(anchors, tf.reshape(idx, [-1]))
