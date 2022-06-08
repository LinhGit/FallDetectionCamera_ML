from __future__ import absolute_import, division, print_function, unicode_literals
import functools
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint


np.set_printoptions(precision=3, suppress=True)


train_data = "E:/Temp/trainn.csv"
test_data = "E:/Temp/test3.csv"
valid_data = "E:/Temp/valid3.csv"


train = pd.read_csv(train_data)


test = pd.read_csv(test_data)

valid = pd.read_csv(valid_data)


LABEL_COLUMN = 'tt'
LABELS = [0, 1]

def get_dataset(file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=12, 
      label_name=LABEL_COLUMN,
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

raw_train_data = get_dataset(train_data)
raw_test_data = get_dataset(test_data)
examples, labels = next(iter(raw_train_data)) # Just the first batch.
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_freatures = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels
NUMERIC_FEATURES = ['v1','v2','v3', 'v4', 'v5']

packed_train_data = raw_train_data.shuffle(500)
packed_test_data = raw_test_data

desc = pd.read_csv(train_data)[NUMERIC_FEATURES].describe()
desc


MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


def normalize_numeric_data(data, mean, std):
  return (data-mean)/std


normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column


numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)



preprocessing_layer = tf.keras.layers.DenseFeatures(numeric_columns)


model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


checkpoint_filepath="E:/Temp/weights-{epoch:02d}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)




train_data = packed_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))


history = model.fit(train_data, 
  epochs=20,
  batch_size=4,
  callbacks=[model_checkpoint_callback])

model.summary()



predictions = model.predict(test_data)


