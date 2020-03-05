













import os
import pickle
import numpy as np
from scipy.io import wavfile
import keras
from test import returnAudio
from keras.models  import Sequential
import matplotlib.pyplot as plt
from keras.regularizers import l1 , l2,l1_l2
from keras.layers import Dropout,Dense,Conv1D, AvgPool1D,GlobalMaxPool1D,MaxPool1D,Flatten, BatchNormalization,GlobalAveragePooling1D,LSTM, Activation
import soundfile as sf

#samples per class to train on
samples = 400
#samples per class to test on
testSamples = 70
#1- SNARE, #2-KICK, #3-808
#
sampleRate = 16000
#total window = sample rate, not always the case
sampleSize = sampleRate

onePath = "snare/train/"
twoPath ="kick/train/"
threePath = "808/train/"

onePathT = "snare/test/"
twoPathT ="kick/test/"
threePathT = "808/test/"


oneTrain = os.listdir(onePath)
twoTrain = os.listdir(twoPath)
threeTrain = os.listdir(threePath)

oneTest = os.listdir(onePathT)
twoTest = os.listdir(twoPathT)
threeTest = os.listdir(threePathT)


oneData = np.empty((0,sampleSize))
twoData = np.empty((0,sampleSize))
threeData = np.empty((0,sampleSize))


oneDataT = np.empty((0,sampleSize))
twoDataT = np.empty((0,sampleSize))
threeDataT = np.empty((0,sampleSize))


oneData = returnAudio(oneTrain,oneData,onePath,sampleSize,samples,sampleRate)
twoData = returnAudio(twoTrain,twoData,twoPath,sampleSize,samples,sampleRate)
threeData = returnAudio(threeTrain,threeData,threePath,sampleSize,samples,sampleRate)
print(oneData.shape)
print(twoData.shape)
print(threeData.shape)


                     #LIST of files,data,path,size of samples, number of samples
oneDataT = returnAudio(oneTest,oneDataT,onePathT,sampleSize,testSamples,sampleRate)
twoDataT = returnAudio(twoTest,twoDataT,twoPathT,sampleSize,testSamples,sampleRate)
threeDataT = returnAudio(threeTest,threeDataT,threePathT,sampleSize,testSamples,sampleRate)

allData = np.vstack((oneData,twoData,threeData))
allDataT = np.vstack((oneDataT,twoDataT,threeDataT))
#############################################
print(oneData.shape)
print(twoData.shape)
print(threeData.shape)

oneLabel = np.full((samples,1),0)
twoLabel = np.full((samples,1),1)
threeLabel = np.full((samples,1),2)

allLabels = np.vstack((oneLabel,twoLabel,threeLabel))
labels = keras.utils.to_categorical(allLabels)
##################################################3
oneLabelT = np.full((testSamples,1),0)
twoLabelT = np.full((testSamples,1),1)
threeLabelT = np.full((testSamples,1),2)

allLabelsT = np.vstack((oneLabelT,twoLabelT,threeLabelT))
labelsT = keras.utils.to_categorical(allLabelsT)

####################################################
num_classes = allLabels.ndim
#labels = labels.reshape(680,4,1)

allData = allData.reshape(samples*3,sampleSize,1)
allDataT = allDataT.reshape(testSamples*3,sampleSize,1)

print(labels.shape)
print(allData.shape)
model = Sequential()

model.add(Conv1D(16,3,input_shape=(sampleSize,1), dilation_rate=1,padding="same"))
model.add(Activation("relu"))
model.add(Conv1D(16,3,padding="same",dilation_rate=2))
model.add(Activation("relu"))
model.add(MaxPool1D(4))
model.add(Dropout(.2))
model.add(Conv1D(32,3,padding="same",dilation_rate=4))
model.add(Activation("relu"))
model.add(Conv1D(32,3,padding="same",dilation_rate=8))
model.add(MaxPool1D(4))
model.add(Activation("relu"))
model.add(Conv1D(64,3, dilation_rate=16,padding="same"))
model.add(Activation("relu"))
model.add(Conv1D(64,3,padding="same",dilation_rate=32))
model.add(Activation("relu"))
model.add(MaxPool1D(4))
model.add(Dropout(.2))
model.add(Conv1D(64,3,padding="same",dilation_rate=64))
model.add(Activation("relu"))
model.add(Conv1D(64,3,padding="same",dilation_rate=128))
model.add(MaxPool1D(4))
model.add(Activation("relu"))
model.add(Dropout(.2))
model.add(GlobalMaxPool1D())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(64,activation="relu"))
model.add(Dropout(.3))
model.add(Dense(3,activation="softmax"))
ada = keras.optimizers.Adadelta(learning_rate=0.2)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
history= model.fit(allData,labels,epochs=40,batch_size=32,validation_data=(allDataT,labelsT),shuffle=True)
score =  model.evaluate(allData,labels,batch_size=32)
print(model.summary())
print(score)
print(history.history.keys())
#plot

fil = open("dil-acc.pkl", "wb")
diction = {"model":history.history['accuracy'],"val":history.history['val_accuracy']}
pickle.dump(diction,fil)
fil.close()

fil = open("dil-loss.pkl", "wb")
diction = {"model":history.history['loss'],"val":history.history['val_loss']}
pickle.dump(diction,fil)
fil.close()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
