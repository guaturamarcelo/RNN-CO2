# %%
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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

# %%
# LSTM MULTVARIATE

#RATIO_TO_PREDICT = ["MP2.5", "MP10", "CO"]
RATIO_TO_PREDICT = "CO"

RATIO_SHIFT = 1

BATCH_SIZE = 64
SEQ_LEN = 72

X_train, y_train = preprocess(train)
X_valid, y_valid = preprocess(valid)
X_test,  y_test  = preprocess(test)

NAME = f"TARGET-{RATIO_TO_PREDICT}_SEQ-{SEQ_LEN}_BATCH_SIZE-{BATCH_SIZE}_SHIFT-{RATIO_SHIFT}_{int(time.time())}"
print(f"\nSTART SEQ_LEN: {SEQ_LEN}, BATCH_SIZE: {BATCH_SIZE}, TARGET: {RATIO_TO_PREDICT}\n")  

#gen_train = TimeseriesGenerator(X_train, y_train, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)
#gen_valid = TimeseriesGenerator(X_valid, y_valid, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)
gen_test  = TimeseriesGenerator(X_test, y_test, SEQ_LEN, sampling_rate=1, batch_size=BATCH_SIZE, shuffle=False)


# %%
# LOAD MODEL

# BEST
best_model = tf.keras.models.load_model(f"models/epochs/RNN_PRED-{RATIO_TO_PREDICT}_BATCH_SIZE-{BATCH_SIZE}_SEQ-{SEQ_LEN}_SHIFT-{RATIO_SHIFT}_Final-09_RMSE-0.087.model")
# WORST
#worst_model = tf.keras.models.load_model(f"training_datas/BKP/models/RNN_Final-10_RMSE-0.075.model")
# %%
# VERSAO COM BATCH_SIZE = 1 BEST
predict = []
real    = []
best = pd.DataFrame()
#
i = 0
while i < (len(gen_test)-1):
    pred = best_model.predict(gen_test[i][0], verbose=0, batch_size=BATCH_SIZE, use_multiprocessing=True, workers=24)
    for j, val in enumerate(pred):
        predict.append(val[0])
        real.append(gen_test[i][1][j][0])
    i = i + 1
best['Real'] = real
best['Predict'] = predict
best[['Predict','Real']].plot(figsize=(10,10))

# %%

# VERSAO COM BATCH_SIZE = 1 BEST
predict = []
real    = []
worst = pd.DataFrame()
#
i = 0
while i < (len(gen_test)-1):
    pred = worst_model.predict(gen_test[i][0], verbose=0, batch_size=BATCH_SIZE, use_multiprocessing=True, workers=24)
    for j, val in enumerate(pred):
        predict.append(val[0])
        real.append(gen_test[i][1][j][0])
    i = i + 1
worst['Real'] = real
worst['Predict'] = predict
worst[['Predict','Real']].plot(figsize=(10,10))