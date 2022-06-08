from __future__ import absolute_import, division, print_function, unicode_literals
import functools
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History

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
NUMERIC_FEATURES = ['v1','v2','v3', 'v4', 'v5', 'v6']

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


checkpoint_filepath="E:/Temp/weights-{epoch:02d}.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='metric',
    mode='max',
    save_best_only=True)




train_data = packed_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))
packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))


history = model.fit(train_data, 
  epochs=1,
  batch_size=4,
  callbacks=[model_checkpoint_callback])
path = "E:/Temp/plp"
model.save_weights("lollo/modeld.h5")


print("Saved model to disk")


model.summary()





import cv2 as cv
import numpy as np
import math
import csv
capture = cv.VideoCapture('F:/OpenCV Project/secondHello/cr0206.avi')
backSub = cv.createBackgroundSubtractorKNN()
x1 = 0
y1 = 0
BG = None
count = 0
rowlist = []
Do_lech_chuan = 0.0
Toc_do_thay_doi_trong_tam = 0.0
mang = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mang1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
k = 0
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    frame1 = backSub.apply(frame)
    frame1 = cv.GaussianBlur(frame1, (5, 5), 2, sigmaY=2)
    thresh = cv.threshold(frame1, 130, 255, cv.THRESH_BINARY)[1]
    kernels = np.ones((4,4), np.uint8)
    kernel = np.array([[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]], dtype=np.uint8)
    thresh = cv.erode(thresh, kernels, iterations=1)
    thresh = cv.dilate(thresh, kernel, iterations=3)
    thresh = cv.erode(thresh, kernels, iterations=1)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if count < 4:
      print("bo hinh")
    else:
        for contour in contours:
          (x, y), (major, minor), angle = cv.fitEllipse(contour)
          print("center x,y:",x,y)
          print("canh dai rong:",major,minor)
          print("goc:",angle)
          if cv.contourArea(contour) < 900:
            continue
          cv.ellipse(frame, ((x,y), (major,minor), angle), (0,255,0), 2)

          x2 = x1 - x
          y2 = y1 - y
          a = pow(y, 2)/pow(x, 2)
          e = math.sqrt(abs(1 - a))
          v0 = math.sqrt(pow(x2, 2) + pow(y2, 2))
          x1 = x
          y1 = y
          print("vantoc", v0)
          print ("e la", e)
          if(k<10):
            mang[k] = angle
            k +=1
          if(k==10):
            for i in range(9):
              Do_lech_chuan = sum(mang)/10
              mang[i] = mang[i+1]
              mang[9] = angle
              print("do lech chuan la:", Do_lech_chuan)

          if(k<10):
            mang1[k] = y
            k +=1
          if(k==10):
            for i in range(9):
              Toc_do_thay_doi_trong_tam = sum(mang1)/10
              mang1[i] = mang1[i+1]
              mang1[9] = y
              print("toc do thay doi trong tam la:", Toc_do_thay_doi_trong_tam)
          rowlist = ["%f" % v0, "%f" % angle, "%f" % x, "%f" % y, "%f" % e , "%f" % Toc_do_thay_doi_trong_tam]
          predict_data=pd.DataFrame(rowlist).to_numpy()
          print(predict_data)
          float(predictions = model.predict(rowlist,batch_size=1))
          print(predictions)          
         
        frame = cv.putText(frame, 'frame: %d' %count, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("ve", frame) 
        cv.imshow('frame',thresh)
            
    count += 1
    if cv.waitKey(30) == 27:
        break

capture.release()
cv.destroyAllWindows()