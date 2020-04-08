# ----------------------------------------------------------------
# https://github.com/mikesmales/Udacity-ML-Capstone/tree/master/Notebooks
# ----------------------------------------------------------------

import librosa 
import librosa.display
from scipy.io import wavfile as wav
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import os

from helpers.wavfilehelper import WavFileHelper


filename = 'UrbanSound8K/audio/fold1/17592-5-0-0.wav' 

librosa_audio, librosa_sample_rate = librosa.load(filename) 
scipy_sample_rate, scipy_audio = wav.read(filename) 

print('Original sample rate:', scipy_sample_rate) 
print('Librosa sample rate:', librosa_sample_rate)

plt.figure(figsize=(12,4))
_ = librosa.display.waveplot(librosa_audio,sr=librosa_sample_rate)
#plt.plot(scipy_audio)
#ipd.Audio(filename)
# plt.show()


mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)
import librosa.display
librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')
# plt.show()


# ----------------------------------------------------------------
# https://github.com/mikesmales/Udacity-ML-Capstone/blob/master/Notebooks/2%20Data%20Preprocessing%20and%20Data%20Splitting.ipynb
# ----------------------------------------------------------------
metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None

    return mfccsscaled


#wavfilehelper = WavFileHelper()
#audiodata = []

features = []

for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath('UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_features(file_name)
    features.append([data, class_label])

    # data = wavfilehelper.read_file_properties(file_name)
    # audiodata.append(data)

print(metadata.head())
print(metadata['class'].value_counts())


# Convert into a Panda dataframe
# audiodf = pd.DataFrame(audiodata, columns=['num_channels','sample_rate','bit_depth'])
# print(audiodf.num_channels.value_counts(normalize=True))
# print(audiodf.bit_depth.value_counts(normalize=True))

featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# ----------------------------------------------------------------
# https://github.com/mikesmales/Udacity-ML-Capstone/blob/master/Notebooks/3%20Model%20Training%20and%20Evaluation.ipynb 
# ----------------------------------------------------------------

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

num_labels = yy.shape[1]
filter_size = 2

# Construct model
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_mlp.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


def print_prediction(file_name):
    prediction_features = np.array([extract_features(file_name)])

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )



file_name = os.path.join(os.path.abspath('UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))

print('Class: Air Conditioner')
filename = 'UrbanSound8K/audio/100852-0-0-0.wav'
print_prediction(filename)

print('Class: Drilling')
filename = 'UrbanSound8K/audio/103199-4-0-0.wav'
print_prediction(filename)

print('Class: Street Music')
filename = 'UrbanSound8K/audio/101848-9-0-0.wav'
print_prediction(filename)

print('Class: Car Horn')
filename = 'UrbanSound8K/audio/100648-1-0-0.wav'
print_prediction(filename)

