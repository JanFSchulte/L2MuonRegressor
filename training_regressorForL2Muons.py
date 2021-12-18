import sys

from tqdm import tqdm
import glob
import uproot
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
#from dataset_arnab import datasets, datasets_dict
#from utils import *
#from tools import *
fig = plt.figure()
ax = fig.add_subplot(111)

import tensorflow
#from tensorflow import constant
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K
import cmsml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
#tensorflow.config.experimental_run_functions_eagerly(True)
opt = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
#opt = Adam(learning_rate=0.0001)
#opt = tensorflow.keras.optimizers.Adam(learning_rate=0.128)
output_path = './'

input_branches = [
'ST_layerID1', 'ST_globalR1', 'ST_globalZ1', 'ST_phi1', 'ST_deltaDir1', 'ST_deltaPhi1','ST_layerID2','ST_globalR2', 'ST_globalZ2', 'ST_phi2', 'ST_deltaDir2', 'ST_deltaPhi2','ST_layerID3','ST_globalR3', 'ST_globalZ3', 'ST_phi3', 'ST_deltaDir3', 'ST_deltaPhi3','ST_layerID4','ST_globalR4', 'ST_globalZ4', 'ST_phi4', 'ST_deltaDir4', 'ST_deltaPhi4'
]

df = pd.read_csv("data/L2Segments_preprocessed.csv")
truth_columns = ['Muon_gen_pt']

prediction_columns = ['pred Muon_gen_pt']
save = True
features = input_branches # imported from tools

def plot_loss(history, label, out_path):
    fig = plt.figure()
    fig.clf()
    plt.rcParams.update({'font.size': 10})
    fig.set_size_inches(5, 4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    out = f'{out_path}/loss_{label}'
    fig.savefig(out)
    print(f'Saved loss plot: {out}')



for i in input_branches:
    print(i)
    print(df[i].max())
    print(df[i].min())
    print(df[i].mean())
    print(df[i].std())

for c in prediction_columns:
    df[c] = -1

df = df.sample(frac=1)

x=df[features].values
y=df[truth_columns].values
print(x)
print(y)

from pickle import dump

scaler = StandardScaler()
# fit scaler on data
scaler.fit(x)
# save the scaler
dump(scaler, open(output_path+'scalerInput.pkl', 'wb'))
# apply transform
x = scaler.transform(x)

outscaler = StandardScaler()
# fit scaler on data
outscaler.fit(y)
# save the scaler
dump(outscaler, open(output_path+'scalerOutput.pkl', 'wb'))
# apply transform
y = outscaler.transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print((x_test)[:,0])
print((x_test)[:,1])





label = f'dnn_L2Regressor_1000Epoch_ScaledInputOutput_y_pred_less'.replace(' ', '_')

input_dim = len(features)
output_dim = len(truth_columns)
#callback = EarlyStopping(monitor='loss', patience=50)
callback = EarlyStopping(monitor='val_loss', patience=50)
inputs = Input(shape=(input_dim,), name=label+'_input')

x = Dense(147, name=label+'_layer_1', activation='relu')(inputs)
x = Dropout(0.0)(x)
x = BatchNormalization()(x)
x = Dense(74, name=label+'_layer_2', activation='relu')(x)
x = Dropout(0.0)(x)
x = BatchNormalization()(x)
x = Dense(37, name=label+'_layer_3', activation='relu')(x)
x = Dropout(0.0)(x)
x = BatchNormalization()(x)
x = Dense(19, name=label+'_layer_4', activation='relu')(x)
x = Dropout(0.0)(x)
x = BatchNormalization()(x)
#x = Dense(16, name=label+'_layer_5', activation='relu')(x)
#x = Dropout(0.2)(x)
#x = BatchNormalization()(x)
#outputs = Dense(output_dim, name=label+'_output',  activation='sigmoid')(x)
outputs = Dense(output_dim, name=label+'_output', activation="linear")(x)

dnn = Model(inputs=inputs, outputs=outputs)
#opt = Adam(learning_rate=0.001)
dnn.compile(
    loss="mse",
    optimizer=opt,
    #run_eagerly=True,
    metrics=["mse"]
)
dnn.summary()

history = dnn.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=256,
    #batch_size=1561,
    callbacks=[callback],
    verbose=0,
    validation_data=(x_test, y_test),
    shuffle=True
)
if save:       
    save_path = f"{label}_full_dataset.pb"
    model_path = f"{label}_L2RegressorModel_training_plots_4layers_lr_0p0001_mse_epochs100"
    dnn.save(model_path)
    print(f'Saving model to {save_path}')
    tensorflow.compat.v1.add_to_collection('features', features)
    cmsml.tensorflow.save_graph(output_path+save_path, dnn, variables_to_constants=True)
    cmsml.tensorflow.save_graph(output_path+save_path+'.txt', dnn, variables_to_constants=True)
plotloss = True
if plotloss:
    plot_loss(history, label, output_path)
y_pred = dnn.predict(x_test)
y_pred[y_pred<0]=0
#y_pred = np.round(y_pred)
prediction = pd.DataFrame(y_pred)
print("L2 pt MSE:%.4f" % mean_squared_error((y_test)[:,0], y_pred[:,0]))

x_ax = range(len(x_test))
print(len((y_test)[:,0]))
print(len(x_test))
print(len(y_pred[:,0]))

plt.clf()
plt.scatter(x_ax, (y_test)[:,0],  s=6, label="L2 pt")
plt.plot(x_ax, y_test[:,0], label="L2 pt prediction")
plt.legend()
plt.savefig(output_path+"final_plot_L2pt"+str(i)+"_dataset.png")

print(prediction)
print(prediction_columns)
#df.to_csv("Output_dataframe_training_newDatasets.csv")

