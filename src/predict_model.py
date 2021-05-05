






# %%
# LOAD MODEL

# BEST
best_model = tf.keras.models.load_model("~/Documents/jupyter/training_datas/BKP/models/RNN_Final-20_RMSE-0.060.model")
# WORST
#worst_model = tf.keras.models.load_model(f"training_datas/BKP/models/RNN_Final-10_RMSE-0.075.model")

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