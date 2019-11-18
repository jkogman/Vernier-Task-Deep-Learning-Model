from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from parameters import parameters
from data_generator import data_generator
import h5py
import time

# General settings

x_max = 10 
x_size, t_size, N_features = parameters(x_max)
batch_size = 64
N_epochs = 10
N_final = 15*N_features                                    
N_final = int(N_final - N_final%(batch_size))
N_val = 100*batch_size
N_test = N_final/5

# Setting up the architecture of the network and compiling

model_delta = Sequential()
model_delta.add(SeparableConv2D(50, (50,30), data_format='channels_first', input_shape=(2,x_size, t_size)))
model_delta.add(MaxPooling2D(pool_size=2, data_format='channels_first'))
model_delta.add(SeparableConv2D(100, (10,6), data_format='channels_first', input_shape=(2,x_size, t_size)))
model_delta.add(MaxPooling2D(pool_size=2, data_format='channels_first'))
model_delta.add(SeparableConv2D(200, (4,3), data_format='channels_first', input_shape=(2,x_size, t_size)))
model_delta.add(MaxPooling2D(pool_size=2, data_format='channels_first'))
model_delta.add(Flatten())
model_delta.add(Dense(100, activation='relu'))
model_delta.add(BatchNormalization(axis=1))
model_delta.add(Activation('relu'))
model_delta.add(Dense(1, activation='sigmoid'))

model_delta.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitiing the model on generated data

filepath="final_model_ep_{epoch:02d}_acc_{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')

start = time.time()
fit_history_del = model_delta.fit_generator(generator = data_generator(batch_size,x_max,'delta','_',100),
                    steps_per_epoch = N_final//batch_size,
                    validation_data = data_generator(batch_size,x_max,'delta','_',100),
                    validation_steps = N_val//batch_size,
                    callbacks = [checkpoint],
                    epochs = N_epochs)
end = time.time()
                    
# Saving history and model

model_delta.save("final_model.h5")
with open("final_model_hist", 'wb') as file_pi:
        pickle.dump(fit_history_del.history, file_pi)

# Evaluating the model

score_del = model_delta.evaluate_generator(generator = data_generator(batch_size,x_max,'delta','_',100), steps = N_test//batch_size)
print(f'Accuracy on testing data, D=100, prior = delta: {score_del[1]}, in {end-start} seconds')