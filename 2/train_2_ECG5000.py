# https://github.com/KristynaPijackova/Tutorials_NNs_and_signals/blob/main/Classification_ECG.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from imblearn.combine import SMOTETomek
import time


def over_under_sampling(dataframe):
    """
    Use SMOTETomek technique to oversample our dataset. 
    
    This function is written to be applied to our datasets, 
    where the first column holds the labels, and the rest is the 
    time sequence. 

    It passes the under-represented data - classes 2-5 along
    with the dominant class 1 into the SMOTETomek over- & undersampler
    to balance the dataset. 
    """
    # lists to store the created values in
    x_res = []
    y_res = []

    for i in range(2,6):

        # create copy of the dataframe
        df_copy = dataframe.copy()
        # choose samples of i-th class
        df = df_copy[df_copy['c0'] == i]
        # add samples from 1st class
        df = df.append(df_copy[df_copy['c0'] == 1])
        # split the dataframe into x - data and y - labels
        x = df.values[:,1:]
        y = df.values[:,0]

        # define the imbalance function
        smtomek = SMOTETomek(random_state=42)
        # fit it on our data
        x_r, y_r = smtomek.fit_resample(x, y)
        
        # we want to skip the data we fit it on - only want the new data
        skip = y.shape[0]
        # append the data into our above lists
        x_res.append(x_r[skip:,:])
        y_res.append(y_r[skip:])

    # return the data as concatenated arrays -> only one array of all samples
    # instead of a list of arrays
    return np.concatenate(x_res), np.concatenate(y_res)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./Figures/2_ECG5000_accuracy.png", format="png")
    return plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./Figures/2_ECG5000_loss.png", format="png")
    return plt.show()

def cm_plot(model, x_test):
    y_predict = model.predict(x_test)
    y_pred = []

    for i in range(len(y_predict)):
        y_pred.append(np.argmax(y_predict[i,:]))

    cm = confusion_matrix(y_test, y_pred)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, cmap='Blues', annot=True, fmt='.2f')
    sns.set(font_scale=1.3)
    plt.title("Confusion Matrix")
    plt.savefig("./Figures/2_ECG5000_confusion.png", format="png")

    return plt.show()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


start_time = time.time()

df_train = pd.read_csv('../datasets/ECG5000/ECG5000_TEST.txt', sep='  ', header=None, engine='python')
df_test = pd.read_csv('../datasets/ECG5000/ECG5000_TRAIN.txt', sep='  ', header=None, engine='python')

df_train = df_train.add_prefix('c')
df_test = df_test.add_prefix('c')

x_os,y_os = over_under_sampling(df_train)

x_train = df_train.values[:,1:]  # [all rows, column 1 to end]
y_train = df_train.values[:,0]   # [all rows, column 0]

x_test = df_test.values[:,1:]    # [all rows, column 1 to end]
y_test = df_test.values[:,0]     # [all rows, column 0]

x_train = np.concatenate((x_train, x_os))
y_train = np.concatenate((y_train, y_os))

scaler = preprocessing.MinMaxScaler()
data_scaler = scaler.fit(x_train)

x_train = data_scaler.transform(x_train)
x_test = data_scaler.transform(x_test)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# we need to index classes from 0 to 4
y_train = y_train - 1            
y_test = y_test - 1


x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=42)

layer_in = layers.Input(shape=(140,1))
layer = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(layer_in)
layer = layers.MaxPool1D(pool_size=2)(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Flatten()(layer)
layer = layers.Dense(32, activation='relu')(layer)
layer_out = layers.Dense(5, activation='softmax')(layer)

model = keras.models.Model(layer_in, layer_out)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

callbacks = [
             keras.callbacks.ModelCheckpoint('ecg_model.h5', 
                                             save_best_only=True, 
                                             monitor='val_loss'),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                               factor=0.1, 
                                               patience=3,
                                               ),
             keras.callbacks.EarlyStopping(monitor='val_loss', 
                                           patience=8,
                                           verbose=1)
             ]

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])


model.summary()

history = model.fit(x_train, y_train,
                    batch_size=64, epochs=100, verbose=2,
                    validation_data=(x_val, y_val),
                    shuffle=True, callbacks=callbacks)

print("Training time:" + str(time.time() - start_time) + " sec")

start_time = time.time()

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)
from sklearn.metrics import classification_report

y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print("Testing time: " + str(x_test.shape[0] / (time.time() - start_time)) + "samples/sec.")

print(classification_report(y_test, y_pred_bool))

plot_acc(history)
plot_loss(history)
cm_plot(model, x_test)