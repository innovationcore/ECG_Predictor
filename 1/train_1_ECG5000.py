# https://github.com/KristynaPijackova/Tutorials_NNs_and_signals/blob/main/Classification_ECG.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from imblearn.combine import SMOTETomek


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
    return plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
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

    return plt.show()




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
# layer = layers.Conv1D(filters=32, kernel_size=16, activation='relu')(layer_in)
# layer = layers.Conv1D(filters=32, kernel_size=16, activation='relu')(layer)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.Conv1D(filters=128, kernel_size=64, activation='relu')(layer_in)
# layer = layers.Conv1D(filters=128, kernel_size=64, activation='relu')(layer)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.Conv1D(filters=256, kernel_size=128, activation='relu')(layer_in)
# layer = layers.Conv1D(filters=256, kernel_size=128, activation='relu')(layer)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.Conv1D(filters=512, kernel_size=256, activation='relu')(layer_in)
# layer = layers.Conv1D(filters=512, kernel_size=256, activation='relu')(layer)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.GlobalAveragePooling1D()(layer)
# layer = layers.Dropout(0.2)(layer)
# layer = layers.Dropout(0.2)(layer)
# #layer_out = layers.Dense(5, activation='softmax')(layer)

layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer_in)
layer = layers.MaxPool1D(pool_size=2)(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer)
layer = layers.MaxPool1D(pool_size=2)(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer_in)
layer = layers.MaxPool1D(pool_size=2)(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer)
layer = layers.MaxPool1D(pool_size=2)(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Flatten()(layer)
layer = layers.Dense(32, activation='relu')(layer)
# layer = layers.Dropout(0.2)(layer)
layer_out = layers.Dense(5, activation='softmax')(layer)

model = keras.models.Model(layer_in, layer_out)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

callbacks = [
             keras.callbacks.ModelCheckpoint('model.h5', 
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

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy", test_acc)
print("Test loss", test_loss)

plot_acc(history)
plot_loss(history)
cm_plot(model, x_test)


# https://github.com/Jarvis-1215/ECG5000/blob/main/ECG_Dataset%20(1).ipynb
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import random
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import layers, losses
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.models import Model
# import os
# import tensorflow as tf
# from tensorflow.compat.v1.keras.backend import set_session
# import torch
# import keras
# from keras import models
# from keras import layers
# from keras.layers import *
# import matplotlib.pyplot as plt
# from keras import backend as K
# import tensorflow as tf
# from keras.models import Sequential


# random.seed(123)

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))

# dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
# raw_data = dataframe.values

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# config = tf.compat.v1.ConfigProto() 
# config.allow_soft_placement=True 
# config.gpu_options.per_process_gpu_memory_fraction = 0.4 
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config = config)
# set_session(sess)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # The last element contains the labels
# labels = raw_data[:, -1]

# # The other data points are the electrocadriogram data
# data = raw_data[:, 0:-1]

# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

# # Normalize to [0, 1]
# min_val = tf.reduce_min(train_data)
# max_val = tf.reduce_max(train_data)

# train_data = (train_data - min_val) / (max_val - min_val)
# test_data = (test_data - min_val) / (max_val - min_val)

# train_data = tf.cast(train_data, tf.float32)
# test_data = tf.cast(test_data, tf.float32)

# x_train = train_data
# x_test = test_data
# y_train = train_labels
# y_test = test_labels

# #validation dataset
# x_val = x_train[:400]
# partial_x_train = x_train[400:]
 
# y_val = y_train[:400]
# partial_y_train = y_train[400:]

# x_train_cnn = np.array(x_train)
# x_test_cnn = np.array(x_test)
# x_val_cnn = np.array(x_val)
# partial_x_train_cnn = np.array(partial_x_train)

# #reshape data to fit model
# x_train_cnn = x_train_cnn.reshape(3998,140,1)
# x_test_cnn = x_test_cnn.reshape(1000,140,1)
# x_val_cnn = x_val_cnn.reshape(400,140,1)
# partial_x_train_cnn = partial_x_train_cnn.reshape(3598,140,1)

# #create model
# layer_in = layers.Input(shape=(140,1))
# layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer_in)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.Conv1D(filters=32, kernel_size=4, activation='relu')(layer)
# layer = layers.MaxPool1D(pool_size=2)(layer)
# layer = layers.Flatten()(layer)
# layer = layers.Dense(32, activation='relu')(layer)
# layer = layers.Dropout(0.2)(layer)
# layer_out = layers.Dense(5, activation='softmax')(layer)

# model = keras.models.Model(layer_in, layer_out)

# # optimizer = keras.optimizers.Adam(learning_rate=0.001)

# #compile model using accuracy to measure model performance
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',f1_m,tf.keras.metrics.AUC()])

# CNN = model.fit(partial_x_train_cnn,
#               partial_y_train,
#               epochs=40,
#               batch_size=32,
#               validation_data=(x_val_cnn,y_val))

# loss_and_metrics_cnn = model.evaluate(x_test_cnn, y_test, batch_size=128)
# classes_cnn = model.predict(x_test_cnn, batch_size=128)

# cnn_folds_acc = []
# cnn_folds_f1 = []
# cnn_folds_auc = []
# for i in range (10):
#     if i ==9:
#         x_test_cnn_folds = x_test_cnn[i*100:]
#         y_test_folds = y_test[i*100:]
#     else:
#         x_test_cnn_folds = x_test_cnn[100*i:100*(i+1)]
#         y_test_folds = y_test[100*i:100*(i+1)]
#     print("testing fold", i+1, "result")
#     test_fold_result = model.evaluate(x_test_cnn_folds, y_test_folds, batch_size=128)
#     cnn_folds_acc.append(test_fold_result[1])
#     cnn_folds_f1.append(test_fold_result[2])
#     cnn_folds_auc.append(test_fold_result[3])
# print(sum(cnn_folds_acc)/len(cnn_folds_acc))
# print(sum(cnn_folds_f1)/len(cnn_folds_f1))
# print(sum(cnn_folds_auc)/len(cnn_folds_auc))

# history_dict_CNN = CNN.history
 
# acc = CNN.history['acc']
# val_acc = CNN.history['val_acc']
# loss = CNN.history['loss']
# val_loss = CNN.history['val_loss']
 
# epochs = range(1, len(acc) + 1)
 
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'r', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('CNN Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
 
# plt.show()

# plt.clf()   # clear figure
# acc_values = history_dict_CNN['acc']
# val_acc_values = history_dict_CNN['val_acc']
# f1_values = history_dict_CNN['f1_m']
# auc_values = history_dict_CNN['auc']
 
# plt.plot(epochs, acc_values, 'r', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.plot(epochs, f1_values, 'y', label='Training F1')
# plt.plot(epochs, auc_values, 'g', label='Training AUC')

# plt.title('CNN Training and Validation Measures Performance')
# plt.xlabel('Epochs')
# plt.ylabel('Index Value')
# plt.legend()

# plt.show()