# https://github.com/KristynaPijackova/Tutorials_NNs_and_signals/blob/main/Classification_ECG.ipynb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
import wfdb
import ast
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report

from clearml import Task, Logger

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

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
        df = pd.concat([df, df_copy[df_copy['c0'] == 1]])
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

def plot_acc(history, out_dir):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    fig_path = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, '1_PTB_accuracy.png'), format="png")
    #plt.show()

def plot_loss(history, out_dir):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    fig_path = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, '1_PTB_loss.png'), format="png")
    #plt.show()

def cm_plot(model, x_test, y_test, out_dir):
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
    fig_path = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(os.path.join(fig_path, '1_PTB_confusion.png'), format="png")
    # Logger.current_logger().report_matplotlib_figure(title='confusion matrix', series='ignored', figure=plt, iteration=None, report_image=False, report_interactive=True)
    #plt.show()

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

def specificity(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../datasets/PTB/', help='Path to the PTB datasets')
    parser.add_argument('--output-dir', type=str, default='results/', help='Path to output stuff')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--val-size', type=float, default=0.2, help='Percent to be used for validation (0.0-1.0)')
    parser.add_argument('--verbosity', type=int, default=1, help='Verbosity (1-3)')
    parser.add_argument('--random-state', type=int, default=69, help='Random state for train_test_split')
    parser.add_argument('--with-clearml',  type=bool, default=False, help='Run training with ClearML enabled')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_opt()
    if opt.with_clearml:
        task = Task.init(project_name='ECG Predictor', task_name='ECG-CNN-LSTM')

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    if not os.path.exists(os.path.join(opt.output_dir, 'figures')):
        os.makedirs(os.path.join(opt.output_dir, 'figures'))
    if not os.path.exists(os.path.join(opt.output_dir, 'logs')):
        os.makedirs(os.path.join(opt.output_dir, 'logs'))

    
    print('[INFO]\t Loading PTB database...')
    path = opt.data_dir
    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                the_class = agg_df.loc[key].diagnostic_class
                if the_class == 'NORM':
                    #tmp.append("HC")
                    tmp.append(0)
                elif the_class == 'MI':
                    #tmp.append("MI")
                    tmp.append(1)
                else:
                    # tmp.append("non-MI")
                    tmp.append(2)
        return tmp

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    val_fold = 9
    
    # Train
    x_train = X[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)]
    y_train = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != val_fold)].diagnostic_superclass.values
    # Validation
    x_val = X[np.where(Y.strat_fold == val_fold)]
    y_val = Y[Y.strat_fold == val_fold].diagnostic_superclass.values
    # Test
    x_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass.values

    labels = []
    remove = []
    for i, y in enumerate(y_train):
        if not y:
            remove.append(i)
        else:
            labels.append(y[0])
    
    x_train = np.delete(x_train, remove, axis=0)
    y_train = np.array(labels)

    labels = []
    remove = []
    for i, y in enumerate(y_val):
        if not y:
            remove.append(i)
        else:
            labels.append(y[0])
    
    x_val = np.delete(x_val, remove, axis=0)
    y_val = np.array(labels)

    labels = []
    remove = []
    for i, y in enumerate(y_test):
        if not y:
            remove.append(i)
        else:
            labels.append(y[0])
    
    x_test = np.delete(x_test, remove, axis=0)
    y_test = np.array(labels)

    #training shapes are (17441, 1000, 12) (number of patient samples, 1000 points in each sample, 12 leads)

    print("[INFO]\t Starting training...")

    # 5 classes in set, but only care about 3
    ## 0 NORM
    ## 1 MI (Myocardial Infarction)
    ## 2 STTC (ST/T Change)
    ## 3 CD (Conduction Distubance)
    ## 4 HYP (Hypertrophy)

    # 3 we care about
    ## HC (Healthy Control)
    ## non-MI (non-Myocardial Infarction)
    ## MI (Myocardial Infarction)


    x_train = np.squeeze(x_train) #removing the None first dimension

    ########### Conv1D and LSTM #############
    layer_in = layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    layer = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(layer_in)
    layer = layers.MaxPool1D(pool_size=2)(layer)
    layer = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(layer)
    # layer = layers.Dropout(0.2)(layer)
    layer = layers.Dense(100, activation='relu')(layer)
    # layer = layers.Dropout(0.2)(layer)
    layer_out = layers.Dense(3, activation='softmax')(layer)

    model = keras.models.Model(layer_in, layer_out)

    optimizer = keras.optimizers.Adam(learning_rate=opt.lr)
    callbacks = [
                keras.callbacks.ModelCheckpoint(os.path.join(opt.output_dir, 'PTBmodel.h5'), 
                                                save_best_only=True, 
                                                monitor='val_loss'),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.1, 
                                                patience=3,
                                                ),
                keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=8,
                                            verbose=1),
                # keras.callbacks.CSVLogger(os.path.join(opt.output_dir, 'logs', 'log'+str(time.time())+'.csv'), 
                #                             append=True, 
                #                             separator=','),
                tf.keras.callbacks.TensorBoard(log_dir=os.path.join(opt.output_dir, 'logs'))
                ]

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer, 
                metrics=['accuracy'])

    model.summary()
    start_time = time.time()

    history = model.fit(x_train, y_train,
                        batch_size=opt.batch_size, epochs=opt.num_epochs, verbose=opt.verbosity,
                        validation_data=(x_val, y_val),
                        shuffle=True, callbacks=callbacks)

    train_time = "Training time: " + str(time.time() - start_time) + " sec"
    print(train_time)

    start_time = time.time()

    test_loss, test_acc = model.evaluate(x_test, y_test)
    test_acc = "Test accuracy: " + str(test_acc)
    print(test_acc)
    test_loss = "Test loss: " + str(test_loss)
    print(test_loss)

    y_pred = model.predict(x_test, batch_size=opt.batch_size, verbose=opt.verbosity)
    y_pred_bool = np.argmax(y_pred, axis=1)

    test_time = "Testing time: " + str(x_test.shape[0] / (time.time() - start_time)) + " samples/sec."
    print(test_time)

    y_pred = y_pred[:, 0]
    speci = specificity(y_test, y_pred) *-1
    print("Specificity: ", speci)
    sensitivity = tf.keras.metrics.SensitivityAtSpecificity(speci)
    print("Sensitivity: ", sensitivity)

    class_report = classification_report(y_test, y_pred_bool)
    print(class_report)
    with open(os.path.join(opt.output_dir, 'test_results.txt'), 'w') as f:
        f.write(train_time + '\n')
        f.write(test_acc + '\n')
        f.write(test_loss + '\n')
        f.write(test_time + '\n')
        f.write(class_report)

    plot_acc(history, opt.output_dir)
    plot_loss(history, opt.output_dir)
    cm_plot(model, x_test, y_test, opt.output_dir)
    
    if opt.with_clearml:
        task.close()


if __name__ == '__main__':
    main()