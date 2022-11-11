import images
import datasets
import models
import losses
import importlib
import metrics as M
import tensorflow as tf
importlib.invalidate_caches(); importlib.reload(images)
importlib.reload(datasets);importlib.reload(models);importlib.reload(losses)
from tensorflow import keras
keras.backend.clear_session()
model = models.distance_model(512)
training_data, validation_data = datasets.create_distance_dataset(batch=1)
epochs = 10
model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), losses.distance_loss)
model.fit(training_data, validation_data=validation_data, epochs=epochs)
