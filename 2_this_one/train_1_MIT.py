import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def cm_plot(cm):
	# Normalise
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots(figsize=(10,10))
	sns.heatmap(cmn, cmap='Blues', annot=True, fmt='.2f')
	sns.set(font_scale=1.3)
	plt.title("Confusion Matrix")
	plt.ylabel('True')
	plt.xlabel('Predicted')
	plt.savefig("./Figures/1_MIT_confusion.png", format="png")
	return plt.show()

def get_model():
    nclass = 5
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    # img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = MaxPool1D(pool_size=2)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)#256
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)#256
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model

if __name__ == '__main__':
    df_train = pd.read_csv("C:\\Users\\sarmstrong\\Desktop\\ECG_Predictor\\datasets\\article2sets\\mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv("C:\\Users\\sarmstrong\\Desktop\\ECG_Predictor\\datasets\\article2sets\\mitbih_test.csv", header=None)

    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

    start_time = time.time()

    model = get_model()
    file_path = "base_mit_model.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=1)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(X, Y, epochs=1000, verbose=1, callbacks=callbacks_list, validation_split=0.1)
    print("Training time: " + str(time.time() - start_time) + "sec.")

    model.load_weights(file_path)
    
    start_time = time.time()

    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    print("Testing time: " + str(X_test.shape[0] / (time.time() - start_time)) + "samples/sec.")

    cm = confusion_matrix(Y_test, pred_test)
    cm_plot(cm)

    f1 = f1_score(Y_test, pred_test, average="macro")

    print("Test f1 score : %s "% f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s "% acc)

    print(classification_report(Y_test, pred_test))