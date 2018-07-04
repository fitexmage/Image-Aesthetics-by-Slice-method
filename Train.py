import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import datetime
import sys
import time
import traceback

import Load as l
import Data as d

# Start running

start_run = time.time() # Record the starting time for running.

print ("Loading...")

# Load training and testing data

try:
    x_train, y_train, valid_train = l.load_train() # Load data from Load.py.
    print ("Training data loaded!")
    x_test, y_test, valid_test = l.load_test()  # Load data from Load.py.
    print ("Testing data loaded!")
except Exception as e:
    f = open(d.error_address, 'a')
    f.write("\n Error: " + "%s" % traceback.format_exc() + "\n\n")
    f.close()
    sys.exit(0)

y_train = d.categorical(y_train) # Make the labels categorical. Only categorical data can be trained.
y_test = d.categorical(y_test) # Make the labels categorical. Only categorical data can be trained.

input_shape = (d.num_rows, d.num_cols, d.channel) # Define the input shape for model.

# Model

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='sigmoid', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(2, 2), activation='sigmoid'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(d.num_classes, activation='softmax'))
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              #optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['mae']) #'accuracy'

# Start training

start_train = time.time() # Record the starting time for training.

model.fit(x_train, y_train,
          batch_size=d.batch_size,
          epochs=d.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Time use

time_run = d.convert_time(time.time() - start_run) # Time for running.
time_train = d.convert_time(time.time() - start_train) # Time for training.
print("It takes " + time_run + " for running.")
print("It takes " + time_train + " for training.")

# Evaluation
score = model.evaluate(x_train, y_train, verbose=0) # Score contains loss, accuracy, and mae.
loss = score[0] # Loss of training
accuracy = "{:.5f}".format(score[1]) # Accuracy of training
print('Test loss: ', loss)
print('Test accuracy: ', accuracy)

# Date
date = datetime.datetime.now()
date = date.strftime('%Y-%m-%d %H:%M:%S')

# Record

write = "Date: " + date + "\n" + \
        "Version: " + d.version + "\n" + \
        "Training data: " + str(d.num_train) + "\n" + \
        "Valid training data: " + str(valid_train) + "\n" + \
        "Testing data: " + str(d.num_test) + "\n" + \
        "Valid testing data: " + str(valid_test) + "\n" + \
        "Time used for running: " + time_run + "\n" + \
        "Time used for training: " + time_train + "\n" + \
        "Accuracy: " + accuracy + "\n"

f = open(d.result_address,'a') # Open the txt file.
f.write("\n" + write) # Write the result to txt.
f.close() # Close the txt file.
print("Result recorded!")

model_json = model.to_json()
with open(d.model_address, "w") as json_file:
    json_file.write(model_json) # Save model, which contains the layers.
model.save_weights(d.parameters_address, overwrite=True) # Save weight.
print("Model saved!")