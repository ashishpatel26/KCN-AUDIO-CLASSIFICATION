import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def returnAudio(dir,target,path,sampleSize,samples,sampleRate):
    count =0
    for sample in dir:
        #add condition to check for wav format

        #Resample wav to desired sample rate
        data, fs = librosa.core.load(path + sample,sr=sampleRate)
        #Resize sample and pad sample to 1 one second long
        #edge handling
        data = librosa.util.fix_length(data,sampleSize)
        if count < samples:
            target = np.vstack((target, data))
            count += 1



    return target


def calcLayers(kernel,sampleSize):
    i =1
    while 1:
        field = 1+2*(kernel-1)*(2**i-1)
        if field>=sampleSize:
            print("RECEPTIVE FIELD IS "+str(field))
            print("AMOUNT OF LAYERS NEEDED WITH KERNEL SIZE "+str(kernel)+" = "+str(i))
            return[field,i]
        i = i+1