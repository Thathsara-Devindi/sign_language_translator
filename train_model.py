import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# data path
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Ayubowan', 'Ow'])
no_sequences = 30
sequence_length = 30

#id labels (Ayubowan = 0, Ow = 1)
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

#collect .npy files again to an array
print(" Loading dataset into memory...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 90% train data , 10% test data )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print(f" Data loaded successfully. Train shape: {X_train.shape}")

# LSTM Neural Network 
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # Output 2 words

#  Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Training the AI model... This might take a moment.")

# train the model (Epochs 2000)
model.fit(X_train, y_train, epochs=2000, batch_size=32, validation_data=(X_test, y_test))

model.summary()

#save model 'action.h5'
model.save('action.h5')
print(" Model trained and saved as 'action.h5' successfully!")