import numpy as np
import pandas as pd
import keras
import os
import sys
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# img = load_img('CHRCC-malignant/CHRCC-105-A10-1.jpg')
# img.width

from keras import layers, initializers

def identity(X, f, filters):

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X

def conv(X, f, filters, s=2):

    F1, F2, F3 = filters

    X_shortcut = X

    X = layers.Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)

    X = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    
    X_shortcut = layers.Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_initializer=initializers.glorot_uniform())(X_shortcut)
    X = layers.BatchNormalization(axis=3)(X_shortcut)

    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X


def create_model(input_shape=(180, 320, 3), classes=1):
    X_input = layers.Input(input_shape)

    X = layers.ZeroPadding2D((3,3))(X_input)

    X = layers.Conv2D(64, (12,12), strides=(3,3), kernel_initializer=initializers.glorot_uniform())(X)
    X = layers.BatchNormalization(axis=3)(X)
    X = layers.Activation('relu')(X)
    X = layers.MaxPooling2D((5,5), strides=(2,2))(X)

    X = conv(X, f=5, filters=[64,64,256], s=1)
    X = identity(X, 3, [64,64,256])

    X = conv(X, f=10, filters=[128, 128, 512], s=3)
    X = identity(X, 10, [128,128,512])
    X = identity(X, 10, [128,128,512])


    X = layers.AveragePooling2D(pool_size=(5,5), padding='same')(X)
    X = layers.Flatten()(X)
    X = layers.Dense(classes, activation='sigmoid', kernel_initializer=initializers.glorot_uniform())(X)

    return keras.Model(X_input, X, name='my_model')



data_path = sys.argv[1]


data = []
file_names = []

split_factor = 2

# Load images
for file in os.listdir(data_path):
    file_names.append(file)
    img = load_img(data_path + file)
    img_array = img_to_array(img)

    print('Loaded ', file, img_array.shape)

    height, width = img_array.shape[0], img_array.shape[1]
    height_new, width_new = height // split_factor, width // split_factor

    # Split image into four and resize to fit the network
    for i in range(2):
        for j in range(2):
            sub_img_array = img_array[height_new * i : height_new * (i+1), width_new * j : width_new * (j+1),:]
            sub_img = array_to_img(sub_img_array).resize((320,180))
            data.append(img_to_array(sub_img))


data = np.array(data)

print('Data shape: ', data.shape)

model = create_model()
model.load_weights('weights.hdf5', by_name=True) # --- 0.855, min -> 0.81/0.96, median 0.92/0.97, mean 0.92/0.985, max -> 0.77/0.82

optimizer = keras.optimizers.RMSprop()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print('Model built')

predictions = model.predict(data, batch_size = 100).reshape(-1)

print('Preditions complete')

predictions_grouped = predictions.reshape(-1,4)


predictions_median = np.apply_along_axis(lambda x: np.mean(x), 1, predictions_grouped)

result_df = pd.DataFrame([file_names, np.rint(predictions_median)]).T
result_df.columns = ['File name', 'Prediction']
result_df.to_csv('Predictions.csv', index=False)
