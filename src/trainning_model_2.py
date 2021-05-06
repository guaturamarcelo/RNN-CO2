# %%
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
pd.options.display.max_rows = 20

# %%
dataset = pd.read_table("../data/Marg_Tiete_Pte_Remedios_pmed.TXT", sep=";")

# 'RADG', 'RADUV', 'UR', 'DV', 'YEAR' * Outras
headers = ['CO', 'HOUR', 'DWEEK', 'YDAY', 'YWEEK', 'NO', 'NO2',
           'MP2.5', 'MP10', 'SO2', 'NOX', 'TEMP', 'VV', 'DV', 'PREC', 'PREC9', 'DATA']
dataset[headers] =  dataset[headers]
dataset['HOUR2'] = pd.to_datetime(dataset['HOUR'], format="%H").astype(str).str[11:16]
dataset['DATE_TIME'] = dataset['DATA'] +' '+ dataset['HOUR2']
dataset['DATE_TIME'] = pd.to_datetime(dataset['DATE_TIME'])
dataset['DATE_TIME'] = dataset['DATE_TIME'].values.astype(np.int64) // 10 ** 9
dataset.set_index('DATE_TIME', inplace=True)
dataset.drop(['DATA', 'HOUR2'], axis=1, inplace=True)
#dataset = dataset.tail(20000)
#print("DATASET DESCRIBE:\n")
#print(dataset.describe().transpose(),"\n")


# %%

#######################
####################### START FUNCTIONS
#######################

def preprocess(data):
  # X
  for i, col in enumerate(data.columns):
    if i == 0:
      df = np.array(data[col])
      df = df.reshape((len(data[col]), 1))
    else:
      val = np.array(data[col])
      val = val.reshape((len(data[col]), 1))
      df = np.hstack([df, val])

  # Y
#  if len(RATIO_TO_PREDICT) == 2:
#    y_target = [RATIO_TO_PREDICT]
#  else:
#    y_target = RATIO_TO_PREDICT

  for i, col in enumerate([RATIO_TO_PREDICT]):
    if i == 0:
      target = np.array(data[col])
      target = target.reshape((len(data[col]), 1))
    else:
      val = np.array(data[col])
      val = val.reshape((len(data[col]), 1))
      target = np.hstack([target, val])

#    target = np.array(data[RATIO_TO_PREDICT])
#    target = target.reshape((len(target), 1))
  return df, target
#
#######################
####################### END FUNCTIONS
#######################

# SCALED DATA
scaler = MinMaxScaler()
scaler.fit(dataset)
datascaled = scaler.transform(dataset)
datascaled = pd.DataFrame(datascaled, columns=dataset.columns)
#datascaled

n = len(datascaled)
train = datascaled[0:int(n*0.8)]
valid = datascaled[int(n*0.8):int(n*0.95)]
test  = datascaled[int(n*0.95):]

# LSTM MULTVARIATE

#RATIO_TO_PREDICT = ["MP2.5", "MP10", "CO"]
RATIO_TO_PREDICT = "CO"

RATIO_SHIFT = 1

X_train, y_train = preprocess(train)
X_valid, y_valid = preprocess(valid)
X_test,  y_test  = preprocess(test)


# %%
for SEQ_LEN in [120, 60, 30]:

   for BATCH_SIZE in [128, 64, 32, 16]:

      NAME = f"TARGET-{RATIO_TO_PREDICT}_SEQ-{SEQ_LEN}_BATCH_SIZE-{BATCH_SIZE}_SHIFT-{RATIO_SHIFT}_{int(time.time())}"

      print(f"\nSTART SEQ_LEN: {SEQ_LEN}, BATCH_SIZE: {BATCH_SIZE}, TARGET: {RATIO_TO_PREDICT}\n")
    
      #######################
      ####################### START PREPROCESS
      #######################

      gen_train = TimeseriesGenerator(X_train, y_train, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)
      gen_valid = TimeseriesGenerator(X_valid, y_valid, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)
      gen_test  = TimeseriesGenerator(X_test, y_test, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)

      for i in range(len(gen_train)):
          x, y = gen_train[i];break
#          print('SHAPE %s => %s' % (x.shape[1:], y.shape));break
#          print('%s => %s' % (x, y));break
      
      #######################
      ####################### END PREPROCESS
      #######################


      #######################
      ####################### START MODEL
      #######################



      EPOCHS = 20  # how many passes through our data
 
      model = Sequential()
      model.add(LSTM(128, input_shape=(x.shape[1:]), return_sequences=True))
      model.add(Dropout(0.2))
      model.add(BatchNormalization())

      model.add(LSTM(128, return_sequences=True))
      model.add(Dropout(0.1))
      model.add(BatchNormalization())

      model.add(LSTM(128))
      model.add(Dropout(0.2))
      model.add(BatchNormalization())

      model.add(Dense(64, activation='linear'))
      model.add(Dropout(0.2))

      model.add(Dense(64, activation='linear'))


      opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=5)

      # Compile model
      model.compile(
          loss=tf.losses.MeanSquaredError(),
          optimizer=opt,
          metrics=[tf.metrics.RootMeanSquaredError()])

      print(model.summary())

      tensorboard = TensorBoard(log_dir=f"../logs/{NAME}")
      filepath = f"RNN_PRED-{RATIO_TO_PREDICT}_BATCH_SIZE-{BATCH_SIZE}_SEQ-{SEQ_LEN}_SHIFT-{RATIO_SHIFT}_"+"Final-{epoch:02d}_RMSE-{root_mean_squared_error:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
      checkpoint = ModelCheckpoint("../models/epochs/{}.model".format(filepath, monitor='val_root_mean_squared_error', verbose=1, save_best_only=True, mode='min')) # saves only the best ones

      history = model.fit(
          gen_train,
          batch_size=1,
          epochs=EPOCHS,
          validation_data=gen_valid,
          callbacks=[tensorboard, checkpoint, callback])

 # %%
      # Score model
      score = model.evaluate(gen_test, verbose=1)
      #print(f'Test loss:', score[0])
      #print(f'Test accuracy:', score[1])
      print(f"\n RESULTADO SEQ_LEN: {SEQ_LEN}, BATCH_SIZE: {BATCH_SIZE}, TARGET: {RATIO_TO_PREDICT}, TEST_LOSS: {score[0]}, TEST_RMSE: {score[1]}")

      # Save model
      model.save(f"../models/final/{NAME}")

