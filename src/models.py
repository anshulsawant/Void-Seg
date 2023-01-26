import tensorflow as tf
from tensorflow import keras
from keras import layers
import datasets
import utils
import losses
from datetime import datetime

class DownBlock(layers.Layer):
  def __init__(self, filters, bn=False):
    super(DownBlock, self).__init__()
    self.conv_1 = layers.Conv2D(filters, 3, padding="same", activation=layers.LeakyReLU())
    self.bn_1 = layers.BatchNormalization() if bn else None
    self.conv_2 = layers.Conv2D(filters, 3, padding="same", activation=layers.LeakyReLU())
    self.bn_2 = layers.BatchNormalization() if bn else None
    self.mp = layers.MaxPooling2D(padding="same", strides=2)

  def call(self, inputs):
    t1 = self.conv_1(inputs)
    t1 = self.bn_1(t1) if self.bn_1 else t1
    t2 = self.conv_2(layers.Concatenate()([inputs, t1]))
    t2 = self.bn_2(t2) if self.bn_2 else t2
    t3 = self.mp(layers.Concatenate()([inputs, t1, t2]))
    return (t1, t2, t3)


class UpBlock(layers.Layer):
    def __init__(self, filters, bn=False):
        super(UpBlock, self).__init__()
        self.conv_1 = layers.Conv2D(filters, 3, padding="same", activation=layers.LeakyReLU())
        self.bn_1 = layers.BatchNormalization() if bn else None
        self.conv_2 = layers.Conv2D(filters, 3, padding="same", activation=layers.LeakyReLU())
        self.bn_2 = layers.BatchNormalization() if bn else None
        self.upsample = layers.Conv2DTranspose(filters//2, 2, 2, padding="same")

    def call(t):
      t1 = self.conv_1(t)
      t1 = bn_1(t1) if bn_1 else t1
      t2 = self.conv_2(layers.concatenate([t, t1]))
      t2 = bn_2(t2) if bn_2 else t2
      t3 = layers.Conv2DTranspose(layers.concatenate([t, t1, t2]))
      return (t1, t2, t3)


class Backbone(layers.Layer):
  def __init__(self):
    super(Backbone, self).__init__()
    self.down_1 = DownBlock(4)
    self.down_2 = DownBlock(8)
    self.down_3 = DownBlock(16)
    self.down_4 = DownBlock(32)
    self.down_5 = DownBlock(64)

  def call(self, inputs):
    _, _, x0 = self.down_1(inputs)
    _, _, x1 = self.down_2(x0)
    _, _, x2 = self.down_3(x1)
    _, _, x3 = self.down_4(x2)
    _, _, x4 = self.down_5(x3)
    return (x1, x2, x3, x4)

class UpsampleAndAdd(layers.Layer):
  def __init__(self):
    super(UpsampleAndAdd, self).__init__()
    self.conv = layers.Conv2D(256, (1,1))
    self.up = layers.Conv2DTranspose(256, 2, 2, padding="same")
    self.add = layers.Add()
  def call(self, x, y):
    x1 = self.conv(x)
    y1 = self.up(y)
    return self.add([x1, y1])

## Feature pyramid layer
class FeaturePyramid(layers.Layer):
  def __init__(self):
    super(FeaturePyramid, self).__init__()
    self.conv_1 = layers.Conv2D(256, (1,1))
    self.ua_1 = UpsampleAndAdd()
    self.ua_2 = UpsampleAndAdd()
    self.ua_3 = UpsampleAndAdd()
    self.conv_2 = layers.Conv2D(256, (3,3), padding="same")
    self.conv_3 = layers.Conv2D(256, (3,3), padding="same")
    self.conv_4 = layers.Conv2D(256, (3,3), padding="same")
    self.conv_5 = layers.Conv2D(256, (3,3), padding="same")
    ## Subsampling
    self.mp = layers.MaxPooling2D(pool_size=(1,1), strides=2)

  def call(self, x1, x2, x3, x4):
    y4 = self.conv_1(x4)
    y3 = self.ua_1(x3, y4)
    y2 = self.ua_2(x2, y3)
    y1 = self.ua_3(x1, y2)
    return (self.conv_2(y1), self.conv_3(y2), self.conv_4(y3), self.conv_5(y4), self.mp(y4))

## Region proposal network
def rpn(anchors_per_location=3, fpn_depth=256):
  ## let feature_map dims be S x S x 256
  feature_map = layers.Input(shape=[None, None, fpn_depth])
  ## shared.shape = (S, S, 512)
  shared = layers.Conv2D(512, (3, 3), padding='same', activation=layers.LeakyReLU(),
                      strides=1, name="rpn_shared")(feature_map)
  # Anchor Score. [batch, height, width, anchors per location * 2].
  # x.shape = (S, S, anchors per location * 2)
  x = layers.Conv2D(2 * anchors_per_location, (1, 1), padding='same',
                    activation="softmax", name="anchor_score")(shared)
  # rpn_probs.shape = (S x S x anchors per location, 2)
  rpn_probs = layers.Reshape((-1, 2))(x)

  # Bounding box refinement. [batch, H, W, anchors per location * 4]
  x = layers.Conv2D(anchors_per_location * 4, (1, 1), padding="same",
              activation=layers.LeakyReLU(), name='rpn_bbox_pred')(shared)

  # rpn_bbox.shape = (S x S x 3, 4)
  rpn_bbox = layers.Reshape((-1, 4))(x)
  outputs = layers.Concatenate(axis=2)([rpn_probs, rpn_bbox])
  return keras.models.Model(feature_map, outputs)

def region_proposal_model(backbone, fpn):
    inputs = layers.Input((None, None, 1))
    (x1, x2, x3, x4) = backbone(inputs)
    (y1, y2, y3, y4, y5) = fpn(x1, x2, x3, x4)
    rpn_model = rpn()
    pb1 = rpn_model(y1)
    pb2 = rpn_model(y2)
    pb3 = rpn_model(y3)
    pb4 = rpn_model(y4)
    pb5 = rpn_model(y5)

    outputs = layers.Concatenate(axis=1, name = "rpn_pb_concat")(
        [pb1, pb2, pb3, pb4, pb5])
    return keras.models.Model(inputs, outputs)

class RpnLoss():
    def loss(rpn_labels, rpn_outputs):
      return tf.map_fn(
          RpnLoss._per_batch_loss, (rpn_labels, rpn_outputs), dtype=(tf.float32, tf.float32),
          fn_output_signature=tf.float32)

    def _per_batch_loss(x):
       rpn_labels = x[0]
       rpn_outputs = x[1]
       anchor_labels = tf.cast(rpn_labels[:,1], dtype=tf.int32)
       positive_indices = tf.where(tf.equal(anchor_labels, 1))
       negative_indices = tf.where(tf.equal(anchor_labels, -1))
       positive_output_indices = tf.cast(
           tf.gather(rpn_labels[:, 0], positive_indices), dtype=tf.int32)
       negative_output_indices = tf.cast(
           tf.gather(rpn_labels[:, 0], negative_indices), dtype=tf.int32)

       label_deltas = tf.gather(rpn_labels[:, 2:6], positive_indices)
       output_deltas = tf.gather(
           rpn_outputs[:,2:6],
           positive_output_indices)

       y_true = tf.concat(
           [tf.ones((tf.shape(positive_indices)[0], 1)),
            tf.zeros((tf.shape(negative_indices)[0], 1))],
           axis = 0)
       output_probs = tf.gather(
           rpn_outputs[:,0:2],
           tf.concat([positive_output_indices, negative_output_indices], axis=0))

       classification_loss = keras.losses.SparseCategoricalCrossentropy()(
           y_true, output_probs)
       regression_loss = keras.losses.Huber()(label_deltas, output_deltas)
       return classification_loss + regression_loss

def build_rpn_model(backbone = None, fpn = None):
  if (not backbone):
    backbone = Backbone()
  if (not fpn):
    fpn = FeaturePyramid()
  rpn = region_proposal_model(backbone, fpn)
  rpn.compile(
      keras.optimizers.Adam(learning_rate = lr), loss = RpnLoss.loss)
  return rpn

def train_rpn(epochs=10, lr=0.0001, model=None, take=None, callbacks = [], batch=4):
  rpn = model
  if (not rpn):
    backbone = Backbone()
    fpn = FeaturePyramid()
    rpn = region_proposal_model(backbone, fpn)
    rpn.compile(
      keras.optimizers.Adam(learning_rate = lr),
      loss = RpnLoss.loss)
  training_data, validation_data = datasets.create_mask_rcnn_dataset(batch=batch)
  if (take):
    training_data = training_data.take(take[0])
    validation_data = training_data.take(take[1])
  rpn.fit(
      training_data,
      validation_data = validation_data,
      epochs=epochs,
      callbacks = callbacks)
  return rpn

def distance_model(size):
  inputs = keras.Input((size, size))
  filters = [4,16,32,64]
  x0 = inputs
  x0 = layers.Reshape((size, size,1))(x0)
  ## x0 = layers.Dropout(0.1) (x0)

  def down_block(filters, t, dropout = 0.2):
    t1 = layers.Conv2D(filters, 5, padding="same", activation="relu") (t)
    t2 = layers.Conv2D(filters, 5, padding="same", activation="relu") (layers.concatenate([t, t1]))
    t3 = layers.MaxPooling2D(padding="valid") (layers.concatenate([t, t1, t2]))
    ## t3 = layers.Dropout(dropout)(t3)
    return t1, t2, t3

  x1, x2, x3 = down_block(4, x0)
  x4, x5, x6 = down_block(16, x3)
  x7, x8, x9 = down_block(32, x6)

  def up_block(filters, t, dropout = 0.2):
    t1 = layers.Conv2D(filters, 5, padding="same", activation="relu") (t)
    t2 = layers.Conv2D(filters, 5, padding="same", activation="relu") (layers.concatenate([t, t1]))
    t3 = layers.Conv2DTranspose(filters//2, 2, 2, padding="valid") (layers.concatenate([t, t1, t2]))
    ## t3 = layers.Dropout(dropout)(t3)
    return (t1, t2, t3)

  x10, x11, x12 = up_block(64, x9)
  x13, x14, x15 = up_block(32, layers.concatenate([x6, x7, x8, x12]))
  x16, x17, x18 = up_block(16, layers.concatenate([x3, x4, x5, x15]))

  x19 = layers.Conv2D(4, 3, activation="relu", padding="same") (
      layers.concatenate([x0, x1, x2, x18]))
  x20 = layers.Conv2D(2, 7, activation="softmax", padding="same") (
      layers.concatenate([x0, x1, x2, x19]))

  y = layers.Lambda(lambda x: x[:,:,:,1]) (x20)
  y = layers.Flatten()(y)
  output = layers.Concatenate()([y, y])
  model = keras.Model(inputs, output)
  return model


def rpn_test():
  keras.backend.clear_session()
  ds = datasets.mask_rcnn_dev_dataset()
  l = models.RpnLoss.loss
  o = keras.optimizers.Adam(learning_rate = 0.0001)
  backbone = models.Backbone()
  fpn = models.FeaturePyramid()
  m = models.region_proposal_model(backbone, fpn)
  for x, y in ds:
    print(x.shape)
    print(y)
    x = tf.expand_dims(x, 0)
    y = tf.expand_dims(y, 0)
    with tf.GradientTape() as tape:
      y_pred = m(x, training=True)
      loss_value = l(y, y_pred)
      grads = tape.gradient(loss_value, m.trainable_weights)
      o.apply_gradients(zip(grads, m.trainable_weights))
