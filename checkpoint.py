import pandas as pd
import numpy as np
import librosa 
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name, e)
        return None

    return mfccsscaled


metadata = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')

features = []

for index, row in metadata.iterrows():
    if (index < 1000):
        file_name = os.path.join(os.path.abspath('UrbanSound8K/audio/'),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        class_label = row["class"]
        data = extract_features(file_name)
        features.append([data, class_label])

print(metadata.head())
print(metadata['class'].value_counts())

featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))


x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# Create the model
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

print("Pre-checkpoint accuracy: %.4f%%" % accuracy)


# Display model architecture summary
model.load_weights('saved_models/weights.best.basic_mlp.hdf5')

# Re-evaluate the model
loss,acc = model.evaluate(x_test,  y_test, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

