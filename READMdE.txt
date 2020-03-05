# Audio Classification using temporal convolutions. 

This is an experiment where I successfully classified drum samples using a Dilated temporal convoltional network with an accuracy of over 97%. 
Spectral features can be expensive and leave out features that may be meaningful. Convoltions are cheap and very efficient.
In this experiment I  tested three architectures, LSTM+CNN,normal temporal CNN, and finally a modified dilated temporal CNN inspired by WaveGAN.
Currently I am writing a paper and completing the final implementation in Pytorch.

 Drum Sample Dataset consists of-->

•	Samples compiled from a personal collection of over 15GB of drum samples used to produce songs. 

•	Three classes tested, Snare, Kick, and 808 bass with 400 training samples, and 70 testing samples for each class.

•	Resampled each drum sample to a sampling rate of 16KHZ

•	Each sample was a .WAV file

•	Samples varied in genre (old school, rap, techno, rock) between training and testing to allow the model to generalize 
between different subtypes of the same class

•	Samples were classified over a one second window. Any samples less than a second were simply zero padded to make all the samples the same size of 16,000 samples.


## Getting Started
You should be able to launch the training cycle for each type of model by just running any of the python files from the terminal(besides test). 
You can modify how your models are saved within each file.
### Prerequisites

You will  need to have these python 3+ packages installed. 
Tensorflow+keras 1.14>
Matplot-lib
soundfile
librosa



## Authors

Kevin Sasso
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

