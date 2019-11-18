from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from parameters import parameters
from data_generator import data_generator
import h5py
import pickle

# General settings

x_max = 10 
x_size, t_size, N_features = parameters(x_max)
batch_size = 64
N_epochs = 10
N_final = 10*N_features                           
N_final = int(N_final - N_final%(batch_size))
N_val = 100*batch_size
N_test = N_final/5

# Setting up the architecture of the network and compiling

model_1 = Sequential()
model_1.add(SeparableConv2D(32, (3,3), activation='relu', padding="same", data_format='channels_first', input_shape=(2,x_size, t_size)))
model_1.add(SeparableConv2D(32, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(MaxPooling2D(pool_size=2, padding='same', data_format='channels_first'))
model_1.add(SeparableConv2D(64, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(SeparableConv2D(64, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(MaxPooling2D(pool_size=2, padding='same', data_format='channels_first'))
model_1.add(SeparableConv2D(128, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(SeparableConv2D(128, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(SeparableConv2D(128, (3,3), activation='relu', padding="same", data_format='channels_first'))
model_1.add(MaxPooling2D(pool_size=2, padding='same', data_format='channels_first'))
model_1.add(Flatten())
model_1.add(Dense(1024, activation='relu'))
model_1.add(Dense(512, activation='relu'))
model_1.add(Dense(256, activation='relu'))
model_1.add(Dense(1, activation='softmax'))

model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitiing the model on generated data (saving every epoch)

filepath="vggnet_based_ep_{epoch:02d}_acc_{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

fit_history_1 = model_1.fit_generator(generator = data_generator(batch_size,x_max,'delta','_',100),
                    steps_per_epoch = N_final//batch_size,
                    validation_data = data_generator(batch_size,x_max,'delta','_',100),
                    validation_steps = N_val//batch_size,
                    callbacks=[checkpoint],
                    epochs = N_epochs)
                                 
# Saving history and model

model_1.save("/content/gdrive/My Drive/Vernier ANN/CNN's/vggnet_based.h5")
with open("/content/gdrive/My Drive/Vernier ANN/CNN's/vggnet_based_hist", 'wb') as file_pi:
        pickle.dump(fit_history_1.history, file_pi)

# Evaluating the model

score = model_1.evaluate_generator(generator = data_generator(batch_size,x_max,'delta','_',100), steps = N_test//batch_size)
print(f'Accuracy on testing data: {score[1]}')
