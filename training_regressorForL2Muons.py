import sys, os
from sys import argv
import glob
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
fig = plt.figure()
ax = fig.add_subplot(111)

import tensorflow as tensorflow
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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers

input_branches = [ 'Muon_L2_deltaPhiFirstLast','Muon_L2_deltaEtaFirstLast',
'ST_globalR1', 'ST_globalZ1', 'ST_phi1', 'ST_eta1', 'ST_deltaPhi1', 'ST_deltaEta1', 'ST_phiBend1', 'ST_etaBend1','ST_globalR2', 'ST_globalZ2', 'ST_phi2', 'ST_eta2', 'ST_deltaPhi2', 'ST_deltaEta2', 'ST_phiBend2', 'ST_etaBend2','ST_globalR3', 'ST_globalZ3', 'ST_phi3', 'ST_eta3', 'ST_deltaPhi3', 'ST_deltaEta3', 'ST_phiBend3', 'ST_etaBend3','ST_globalR4', 'ST_globalZ4', 'ST_phi4', 'ST_eta4', 'ST_phiBend4', 'ST_etaBend4'
]


df = pd.read_csv("data/L2Segments_preprocessed.csv")
truth_columns = ['Muon_gen_pt','Muon_gen_eta','Muon_gen_phi']

prediction_columns = ['pred Muon_gen_pt']
save = True
features = input_branches # imported from tools


learningRate = float(argv[1])
epochs = int(argv[2])
activation = str(argv[3])

print ("Running with learning rate %f, epochs %d, activiation function %s"%(learningRate,epochs,activation))

output_path = 'L2DNN_trainingPlotsNew_ptEtaPhi_learnRate_%f_epochs_%d_activation_%s_4Layers/'%(learningRate,epochs,activation)

if not os.path.exists(output_path):
	os.makedirs(output_path)


print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))


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
    out = '%s/loss_%s.pdf'%(output_path,label)
    fig.savefig(out)
    print('Saved loss plot: %s'%out)

ranges = {

	"Muon_gen_pt": [0,1000],
	"Muon_gen_eta": [-3,3],
	"Muon_gen_phi": [-3.14,3.14],
	"Muon_L2_deltaPhiFirstLast": [0,0.3],
	"Muon_L2_deltaEtaFirstLast": [0,0.3],
	"Muon_L2_deltaDirFirstLast": [-1,1],
	"ST_layerID1": [-0.5,12.5],
	"ST_globalR1": [0,800],
	"ST_globalZ1": [0,1000],
	"ST_phi1": [-3.14,3.14],
	"ST_eta1": [-2.4,2.4],
	"ST_deltaPhi1": [0,0.3],
	"ST_deltaEta1": [0,0.3],
	"ST_deltaDir1": [-1,1],
	"ST_phiBend1": [0,0.05],
	"ST_etaBend1": [0,0.05],
	"ST_layerID2": [-0.5,12.5],
	"ST_globalR2": [0,800],
	"ST_globalZ2": [0,1000],
	"ST_phi2": [-3.14,3.14],
	"ST_eta2": [-2.4,2.4],
	"ST_deltaPhi2": [0,0.3],
	"ST_deltaEta2": [0,0.3],
	"ST_deltaDir2": [-1,1],
	"ST_phiBend2": [0,0.05],
	"ST_etaBend2": [0,0.05],
	"ST_layerID3": [-0.5,12.5],
	"ST_globalR3": [0,800],
	"ST_globalZ3": [0,1000],
	"ST_phi3": [-3.14,3.14],
	"ST_eta3": [-2.4,2.4],
	"ST_deltaPhi3": [0,0.3],
	"ST_deltaEta3": [0,0.3],
	"ST_deltaDir3": [-1,1],
	"ST_phiBend3": [0,0.05],
	"ST_etaBend3": [0,0.05],
	"ST_layerID4": [-0.5,12.5],
	"ST_globalR4": [0,800],
	"ST_globalZ4": [0,1000],
	"ST_phi4": [-3.14,3.14],
	"ST_eta4": [-2.4,2.4],
	"ST_phiBend4": [0,0.05],
	"ST_etaBend4": [0,0.05],
}

nBins = {

	"Muon_gen_pt": 50,
	"Muon_gen_eta": 50,
	"Muon_gen_phi": 50,
	"Muon_L2_deltaPhiFirstLast": 50,
	"Muon_L2_deltaEtaFirstLast": 50,
	"Muon_L2_deltaDirFirstLast": 50,
	"ST_layerID1": 12,
	"ST_globalR1": 20,
	"ST_globalZ1": 20,
	"ST_phi1": 20,
	"ST_eta1": 20,
	"ST_deltaPhi1": 50,
	"ST_deltaEta1": 50,
	"ST_deltaDir1": 50,
	"ST_phiBend1": 50,
	"ST_etaBend1": 50,
	"ST_layerID2": 12,
	"ST_globalR2": 20,
	"ST_globalZ2": 20,
	"ST_phi2": 20,
	"ST_eta2": 20,
	"ST_deltaPhi2": 50,
	"ST_deltaEta2": 50,
	"ST_deltaDir2": 50,
	"ST_phiBend2": 50,
	"ST_etaBend2": 50,
	"ST_layerID3": 12,
	"ST_globalR3": 20,
	"ST_globalZ3": 20,
	"ST_phi3": 20,
	"ST_eta3": 20,
	"ST_deltaPhi3": 50,
	"ST_deltaEta3": 50,
	"ST_deltaDir3": 50,
	"ST_phiBend3": 50,
	"ST_etaBend3": 50,
	"ST_layerID4": 12,
	"ST_globalR4": 20,
	"ST_globalZ4": 20,
	"ST_phi4": 20,
	"ST_eta4": 20,
	"ST_phiBend4": 50,
	"ST_etaBend4": 50,
}

def plotInput(out, values, label):

	plt.clf()
	#plt.hist(values, nBins, range=[6.5, 12.5],  label=label)
	plt.hist(values, nBins[label], range=ranges[label],  label=label)
	#plt.plot(y_pred, y_test, label="gen pt prediction")
	#plt.legend()
	plt.xlabel(label)
	plt.ylabel("N")
	plt.savefig(out+"/inputDistribution_%s.png"%label)
	plt.savefig(out+"/inputDistribution_%s.pdf"%label)


def plotInputScaled(out, values, label):

	plt.clf()
	#plt.hist(values, nBins, range=[6.5, 12.5],  label=label)
	plt.hist(values, nBins[label],  label=label)
	#plt.plot(y_pred, y_test, label="gen pt prediction")
	#plt.legend()
	plt.xlabel(label)
	plt.ylabel("N")
	plt.savefig(out+"/inputDistributionScaled_%s.png"%label)
	plt.savefig(out+"/inputDistributionScaled_%s.pdf"%label)





for i in input_branches:
	plotInput(output_path,df[i],i)
for i in truth_columns:
	plotInput(output_path,df[i],i)



inputRanges = {}

for i in input_branches:

    inputRanges[i] = {}
    inputRanges[i]["mean"] = df[i].mean()
    inputRanges[i]["std"] = df[i].std()
    inputRanges[i]["min"] = df[i].min()
    inputRanges[i]["max"] = df[i].max()


outfile = open("inputRanges.json",'w')
json.dump(inputRanges,outfile)
outfile.close()


outputRanges = {}

for i in truth_columns:

    outputRanges[i] = {}
    outputRanges[i]["mean"] = df[i].mean()
    outputRanges[i]["std"] = df[i].std()
    outputRanges[i]["min"] = df[i].min()
    outputRanges[i]["max"] = df[i].max()

outfile = open("outputRanges.json",'w')
json.dump(outputRanges,outfile)
outfile.close()



x=df[features].values
y=df[truth_columns].values

from pickle import dump

scale=True

if scale:
	scaler = MinMaxScaler()
	# fit scaler on data
	scaler.fit(x)
	# save the scaler
	dump(scaler, open(output_path+'scalerInput.pkl', 'wb'))
	# apply transform
	x = scaler.transform(x)

	outscaler = MinMaxScaler()
	# fit scaler on data
	outscaler.fit(y)
	# save the scaler
	dump(outscaler, open(output_path+'scalerOutput.pkl', 'wb'))
	# apply transform
	y = outscaler.transform(y)


for index, i in enumerate(input_branches):
	plotInputScaled(output_path,x[:index],i)
for index, i in enumerate(truth_columns):
	plotInputScaled(output_path,y[:index],i)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)
label = 'dnnTest_L2Regressor_pTEtaPhi_learnRate_%f_epoch_%d_activation_%s'%(learningRate,epochs,activation)

input_dim = len(features)
output_dim = len(truth_columns)
callback = EarlyStopping(monitor='loss', patience=50)
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
outputs = Dense(output_dim, name=label+'_output',  activation=activation)(x)


opt = tensorflow.keras.optimizers.Adam(learning_rate=learningRate)

dnn = Model(inputs=inputs, outputs=outputs)
dnn.compile(
    loss="mse",
    optimizer=opt,
    metrics=["mse"]
)
dnn.summary()

history = dnn.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=256,
    callbacks=[callback],
    verbose=0,
    validation_data=(x_test, y_test),
    shuffle=True
)




if save:       
    save_path = "%s_full_dataset.pb"%label
    model_path = "%s_L2RegressorModel_training_4layers"%label
    dnn.save(model_path)
    tensorflow.compat.v1.add_to_collection('features', features)
    cmsml.tensorflow.save_graph(output_path+save_path, dnn, variables_to_constants=True)
    cmsml.tensorflow.save_graph(output_path+save_path+'.txt', dnn, variables_to_constants=True)
plotloss = True
if plotloss:
    plot_loss(history, label, output_path)
y_pred = dnn.predict(x_test)
prediction = pd.DataFrame(y_pred)
print("L2 pt MSE:%.4f" % mean_squared_error((y_test)[:,0], y_pred[:,0]))

x_ax = range(len(x_test))

plt.clf()
plt.scatter(x_ax, (y_test)[:,0],  s=6, label="L2 pt")
plt.plot(x_ax, y_test[:,0], label="L2 pt prediction")
plt.legend()
plt.savefig(output_path+"final_plot_L2pt"+str(i)+"_dataset.png")
plt.savefig(output_path+"final_plot_L2pt"+str(i)+"_dataset.pdf")

print ("predcited")
print (y_pred)

for index, i in enumerate(truth_columns):
	print (y_test[:,index])
	#plt.scatter(y_test[:,index], y_pred[:,index],  s=6, label="%s vs prediction"%i)

	heatmap, xedges, yedges = np.histogram2d(y_test[:,index], y_pred[:,index], bins=50)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	plt.clf()
	plt.imshow(heatmap.T,  cmap='jet', extent=extent, origin='lower')
	plt.xlabel("true %s"%i)
	plt.ylabel("predicted %s"%i)
	plt.savefig(output_path+"final_plot_"+str(i)+"_dataset.png")
	plt.savefig(output_path+"final_plot_"+str(i)+"_dataset.pdf")




