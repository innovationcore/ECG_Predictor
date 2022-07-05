# https://github.com/KristynaPijackova/Tutorials_NNs_and_signals/blob/main/Classification_ECG.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
import wfdb as wf
import h5py
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


def over_under_sampling(dataframe, labels):
    """
    Use SMOTETomek technique to oversample our dataset. 
    
    This function is written to be applied to our datasets, 
    where the first column holds the labels, and the rest is the 
    time sequence. 

    It passes the under-represented data - classes 2-5 along
    with the dominant class 1 into the SMOTETomek over- & undersampler
    to balance the dataset. 
    """
    print("Starting Over Under Fitting...")
    # lists to store the created values in
    x_res = []
    y_res = []

    for i in range(1,5):
        # create copy of the dataframe
        df_copy = dataframe.copy()
        df_copy['labels'] = labels

        # choose samples of i-th class
        df = df_copy[df_copy['labels'] == i]
        # add samples from 1st class
        df = pd.concat((df, df_copy[df_copy['labels'] == 0]))
        #df = df.append(df_copy[df_copy['labels'] == 0])
        # split the dataframe into x - data and y - labels
        x = df.values[:,:-1]
        y = df.values[:,-1]
        print(x.shape, y.shape)

        num_neighbors = 5
        isLowSamples = True
        while isLowSamples and num_neighbors > 0:
            try:
                # define the imbalance function
                smote = SMOTE(k_neighbors=num_neighbors)
                smtomek = SMOTETomek(smote=smote, random_state=42)
                # fit it on our data
                x_r, y_r = smtomek.fit_resample(x, y)
                # we want to skip the data we fit it on - only want the new data
                skip = y.shape[0]
                # append the data into our above lists
                x_res.append(x_r[skip:,:])
                y_res.append(y_r[skip:])
                isLowSamples = False
            except:
                print("Number of samples too low, retrying with less neighbors.")
                num_neighbors = num_neighbors - 1

    # return the data as concatenated arrays -> only one array of all samples
    # instead of a list of arrays
    print("Over Under Fitting... done.")
    return np.concatenate(x_res), np.concatenate(y_res)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./Figures/1_MIT_accuracy.png", format="png")
    return plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    plt.savefig("./Figures/1_MIT_loss.png", format="png")
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
    plt.savefig("./Figures/1_MIT_confusion.png", format="png")

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

def Load_Data(File):
	File = h5py.File(f'{File}.hdf5', 'r')
	Normal = File['Normal']
	Sup = File['Sup']
	Ven = File['Ven']
	Fusion = File['Fusion']
	Unknown = File['Unknown']
	return Normal, Sup, Ven, Fusion, Unknown

def create_labels(N, S, V, F, U):
	N_Labels = np.zeros(len(N), dtype=int)
	S_Labels = np.ones(len(S), dtype=int)
	V_Labels = np.full((len(V)), 2, dtype=int)
	F_Labels = np.full((len(F)), 3, dtype=int)
	U_Labels = np.full((len(U)), 4, dtype=int)

	N_Labels = np.reshape(N_Labels, (-1, 1))
	S_Labels = np.reshape(S_Labels, (-1, 1))
	V_Labels = np.reshape(V_Labels, (-1, 1))
	F_Labels = np.reshape(F_Labels, (-1, 1))
	U_Labels = np.reshape(U_Labels, (-1, 1))

	return N_Labels, S_Labels, V_Labels, F_Labels, U_Labels

def preprocessing_f(Dimension=None, File=None):
	N, S, V, F, U = Load_Data(File=File)
	NSet, SSet, VSet, FSet, USet = N[:], S[:], V[:], F[:], U[:]
	N_L, S_L, V_L, F_L, U_L = create_labels(NSet, SSet, VSet, FSet, USet)

	if Dimension is not None:
		N_L = np.reshape(N_L, (-1, 1))
		S_L = np.reshape(S_L, (-1, 1))
		V_L = np.reshape(V_L, (-1, 1))
		F_L = np.reshape(F_L, (-1, 1))
		U_L = np.reshape(U_L, (-1, 1))
	Dataset = np.concatenate((NSet, SSet, VSet, FSet, USet), axis=0)
	Labels = np.concatenate((N_L, S_L, V_L, F_L, U_L), axis=0)

	return Dataset, Labels

start_time = time.time()

Dataset, Labels = preprocessing_f(File='../datasets/MIT-BIH/train')

x_train, x_val, y_train, y_val = train_test_split(Dataset, Labels, test_size=0.2)

# df_train = pd.DataFrame(x_train)
# x_train = df_train.add_prefix('c')
# df_test = pd.DataFrame(y_train)
# y_train = df_test.add_prefix('c')
# df_val0 = pd.DataFrame(x_val)
# x_val = df_val0.add_prefix('c')
# df_val1 = pd.DataFrame(y_val)
# y_val = df_val1.add_prefix('c')


# x_os,y_os = over_under_sampling(x_train, y_train)
# x_val_os, y_val_os = over_under_sampling(x_val, y_val)

# x_train = np.concatenate((x_train, x_os))
# y_train = np.concatenate((y_train, np.expand_dims(y_os, axis=1)))
# x_val = np.concatenate((x_val, x_val_os))
# y_val = np.concatenate((y_val, np.expand_dims(y_val_os, axis=1)))

scaler = preprocessing.MinMaxScaler()
data_scaler_train = scaler.fit(x_train)
data_scaler_val = scaler.fit(x_val)

x_train = data_scaler_train.transform(x_train)
x_train = np.expand_dims(x_train, axis=-1)
x_val = data_scaler_val.transform(x_val)
x_val = np.expand_dims(x_val, axis=-1)


layer_in = layers.Input(shape=(x_train.shape[1],1))
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
layer_out = layers.Dense(5, activation='softmax')(layer)

model = keras.models.Model(layer_in, layer_out)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

callbacks = [
             keras.callbacks.ModelCheckpoint('mit_model.h5', 
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

print("Training time: " + str(time.time() - start_time) + "sec.")


start_time = time.time()

model.load_weights('./mit_model.h5')
path = '../datasets/MIT-BIH/test'
x_test, y_test = preprocessing_f(File=path)
data_scaler_test = scaler.fit(x_test)
x_test = data_scaler_test.transform(x_test)
x_test = np.expand_dims(x_test, axis=-1)
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

y_pred = model.predict(x_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)


print("Testing time: " + str(x_test.shape[0] / (time.time() - start_time)) + "samples/sec.")

print(classification_report(y_test, y_pred_bool))

plot_acc(history)
plot_loss(history)
cm_plot(model, x_test)