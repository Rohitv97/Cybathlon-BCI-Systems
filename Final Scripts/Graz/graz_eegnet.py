import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras import layers, Model, models
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from graz_augment import augment_data


def load_data(mode='train', fno = 1):
    """
    Loading Data
    :params: mode = Train by default. Specifies which session data to load for a particular participant
    :params: fno = Participant number
    :return: Return the required EEG data
    """
    if fno != 4:
        if (mode=='train'):
            fname = '/content/drive/My Drive/BCI/graz/A0' + str(fno) + 'T.mat'
            file_data = loadmat(fname)
            data = file_data['data']
            df = pd.DataFrame()
            for i in range(0, 6):
                idx = 3+i
                pos_data = data[0][idx][0][0][1]
                label_data = data[0][idx][0][0][2]
                temp = pd.DataFrame(data[0][idx][0][0][0])
                label = np.zeros(len(temp))
                count = 0
                for j in pos_data:
                    label[j] = label_data[count]
                    count += 1
                temp['class'] = label
                df = pd.concat([df, temp], ignore_index=True)

        elif (mode=='test'):
            fname = '/content/drive/My Drive/BCI/graz/A0' + str(fno) + 'E.mat'
            file_data = loadmat(fname)
            data = file_data['data']
            df = pd.DataFrame()
            for i in range(0, 6):
                idx = 3+i
                pos_data = data[0][idx][0][0][1]
                label_data = data[0][idx][0][0][2]
                temp = pd.DataFrame(data[0][idx][0][0][0])
                label = np.zeros(len(temp))
                count = 0
                for j in pos_data:
                    label[j] = label_data[count]
                    count += 1
                temp['class'] = label
                df = pd.concat([df, temp], ignore_index=True)
        
        return df

    else:
        if (mode=='train'):
            fname = '/content/drive/My Drive/BCI/graz/A0' + str(fno) + 'T.mat'
            file_data = loadmat(fname)
            data = file_data['data']
            df = pd.DataFrame()
            for i in range(0, 6):
                idx = 1+i
                pos_data = data[0][idx][0][0][1]
                label_data = data[0][idx][0][0][2]
                temp = pd.DataFrame(data[0][idx][0][0][0])
                label = np.zeros(len(temp))
                count = 0
                for j in pos_data:
                    label[j] = label_data[count]
                    count += 1
                temp['class'] = label
                df = pd.concat([df, temp], ignore_index=True)

        elif (mode=='test'):
            fname = '/content/drive/My Drive/BCI/graz/A0' + str(fno) + 'E.mat'
            file_data = loadmat(fname)
            data = file_data['data']
            df = pd.DataFrame()
            for i in range(0, 6):
                idx = 1+i
                pos_data = data[0][idx][0][0][1]
                label_data = data[0][idx][0][0][2]
                temp = pd.DataFrame(data[0][idx][0][0][0])
                label = np.zeros(len(temp))
                count = 0
                for j in pos_data:
                    label[j] = label_data[count]
                    count += 1
                temp['class'] = label
                df = pd.concat([df, temp], ignore_index=True)
            
        return df

def band_pass_filter(eeg, freq_range):
    """
    Band Pass Filtering
    :params: eeg = EEG data
    :params: freq_range = Frequency range for band-pass filtering
    :return: Filtered data
    """
    info = mne.create_info(22, 250, ch_types=["eeg"] * 22)
    raw = mne.io.RawArray(eeg.T, info)
    raw.filter(freq_range[0], freq_range[1], fir_design='firwin')

    return raw._data.T

def drop_classes(df):
    """
    Drop un-required classes if any
    :params: df = EEG data
    :return: EEG data after dropping classes
    """
    i = 0
    indexes_to_drop = []
    while i < len(df):
        if(df['class'][i]==4):
            list2 = list(range(i, 1001+i))
            i += 1001
            indexes_to_drop.extend(list2)
        elif(df['class'][i]==3):
            list2 = list(range(i, 1001+i))
            i += 1001
            indexes_to_drop.extend(list2)
        else:
            i += 1

    indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
    df_sliced = df.take(list(indexes_to_keep))

    df_sliced = df_sliced.reset_index(drop=True)
    return df_sliced

def convert_class(sfreq, trigger_points, df):
    """
    Generates a list of asynchronous class labels for the dataset
    :params: sfreq = Sampling Frequency
    :params: trigger_points = A dictionary which denotes the time period of each class/label
    :params: df = EEG data
    :return: asynch_label = A numpy array of the labels for the data
    """
    asynch_label = []
    i = 0
    while i < len(df['class']):
        if(df['class'][i] == 1):
            end = (sfreq * trigger_points[1]) + 1
            for j in range(i, i+end):
                asynch_label.append(1)
            i += end
        elif(df['class'][i] == 2):
            end = (sfreq * trigger_points[2]) + 1
            for j in range(i, i+end):
                asynch_label.append(2)
            i += end
        elif(df['class'][i] == 3):
            end = (sfreq * trigger_points[3]) + 1
            for j in range(i, i+end):
                asynch_label.append(3)
            i += end
        elif(df['class'][i] == 4):
            end = (sfreq * trigger_points[4]) + 1
            for j in range(i, i+end):
                asynch_label.append(4)
            i += end
        elif(df['class'][i] == 0):
            asynch_label.append(0)
            i += 1

    return np.array(asynch_label)

def prune_records(data_new, label):
    """
    Remove from EEG data all classes which are 0. These are irrelevant
    :params: data_new = EEG data
    :params: label = Trigger points for EEG data
    :return: New train_data which is EEG data without 0 class
    :return: New label_data which is triggers without 0 trigger
    """
    train_data = []
    label_data = []

    for i in range(len(label)):
        if label[i] != 0:
            label_data.append(label[i])
            train_data.append(data_new[i])
    label_data = np.array(label_data)
    train_data = np.array(train_data)

    return train_data, label_data

def data_win(sfreq, data, asynch_label):
    """
    Window the EEG data to make it asynchronous
    :params: sfreq = Sampling frequency
    :params: data = EEG data
    :params: asynch_label = Asynchronous label for the EEG data
    :return: Windowed data whose shape is 3D (batch_size, samples, channels)
    :return: Labels corresponding to the windowed data
    """
    sampling_window = 2 * sfreq
    shift_length = 1 * sfreq
    t_start = 0

    new_data = []
    labels = []


    while t_start + sampling_window < data.shape[0]:
        new_data.append(data[t_start:t_start+sampling_window, :].T)
        labels.append(asynch_label[t_start:t_start+sampling_window])

        t_start = t_start + shift_length
    return np.array(new_data), np.array(labels)

def transform_label(label_new):
    """
    Converted asynchronous windowed labels back to trigger points
    :params: label_new = windowed label
    :return: new label which is the new trigger points for windowed data
    """
    label = []
    for i in label_new:
        count1 = np.count_nonzero(i==1)
        count2 = np.count_nonzero(i==2)
        count3 = np.count_nonzero(i==3)
        count4 = np.count_nonzero(i==4)
        if count1 >= 250:
            to_add = 1
        elif count2 >= 250:
            to_add = 2
        elif count3 >= 250:
            to_add = 3
        elif count4 >= 250:
            to_add = 4
        else:
            to_add = 0
        label.append(to_add)
        
    label = np.array(label)
    return label

def get_train_data(fno=1):
    """
    Function to load and preprocess the training data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}
    freq_range = [8, 32]

    to_augment = True #False
    #True if you want to augment the dataset, else set it to False

    df = load_data(mode='train', fno=fno)
    train_df = drop_classes(df)
    train_df = train_df.drop([22, 23, 24], axis=1)
    asynch_label = convert_class(sfreq=sfreq, trigger_points=trigger_points, df=train_df)

    if to_augment == True:
        augmented_data = augment_data(train_df, asynch_label, n_times=100, split_size=2)
        extra_label = augmented_data[:, -1]
        augmented_data = pd.DataFrame(augmented_data)
        augmented_data = augmented_data.rename(columns = {22: 'class'})
        train_df = pd.concat([train_df, augmented_data])
        asynch_label = np.concatenate((asynch_label, extra_label))

    out_data = band_pass_filter(train_df.iloc[:, :-1].values, freq_range=freq_range)

    X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
    y = transform_label(y)
    X, y = prune_records(X, y)
    dim1, dim2, dim3 = X.shape
    X_new = X.reshape((dim1, 1, dim2, dim3))
    y = y-1

    return X_new, y


def get_eval_data(fno=1):
    """
    Function to load and preprocess the testing data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}
    freq_range = [8, 32]

    df = load_data(mode='test', fno=fno)
    train_df = drop_classes(df)
    train_df = train_df.drop([22, 23, 24], axis=1)
    asynch_label = convert_class(sfreq=sfreq, trigger_points=trigger_points, df=train_df)

    out_data = band_pass_filter(train_df.iloc[:, :-1].values, freq_range=freq_range)

    X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
    y = transform_label(y)
    X, y = prune_records(X, y)
    dim1, dim2, dim3 = X.shape
    X_eval = X.reshape((dim1, 1, dim2, dim3))
    y = y - 1

    return X_eval, y

def EEGNET(channels=22, samples=500):
    """
    Implementing EEGNet Architecture
    """
    input1 = layers.Input(shape=(1, channels, samples))
    b1 = layers.Conv2D(8, (1, 250), padding='same', use_bias=False, data_format='channels_first')(input1)
    b1 = layers.BatchNormalization(axis=1)(b1)
    b1 = layers.DepthwiseConv2D((channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), data_format='channels_first')(b1)
    b1 = layers.BatchNormalization(axis=1)(b1)
    b1 = layers.Activation('elu')(b1)
    b1 = layers.AveragePooling2D((1, 4), data_format='channels_first')(b1)
    b1 = layers.Dropout(0.5)(b1)

    b2 = layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False, data_format='channels_first')(b1)
    b2 = layers.BatchNormalization(axis=1)(b2)
    b2 = layers.Activation('elu')(b2)
    b2 = layers.AveragePooling2D((1, 8), data_format='channels_first')(b2)
    b2 = layers.Dropout(0.5)(b2)

    flatten = layers.Flatten()(b2)

    dense = layers.Dense(1, kernel_constraint=max_norm(0.25))(flatten)
    activation = layers.Activation('sigmoid')(dense)

    eegnet = Model(inputs = input1, outputs=activation)

    return eegnet

def network_train(X_train, y_train, X_eval, y_eval):
    """
    Trains the network multiple times and takes average results
    :params: X = training features
    :params: y = training labels
    :params: X_eval = Evluation data set
    :params: y_eval = Evaluation labels
    """
    X = X_train
    y = y_train
    print(X_train.shape)
    dim1, dim2, dim3, dim4 = X_train.shape
    split_idx = ((75 * dim1) // 100) + 1
    X_train = X[:split_idx, :, :, :]
    y_train = y[:split_idx]
    X_test = X[split_idx:, :, :, :]
    y_test = y[split_idx:]
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    sum_train_acc = []
    sum_test_acc = []
    sum_train_kappa = []
    sum_test_kappa = []

    ba = BinaryAccuracy()
    kappa = CohenKappa(num_classes=2)

    times = 50

    for i in range(times):
        eegnet = EEGNET(channels=22, samples=500)
        eegnet.summary()

        my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        ]

        eegnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=[ba, kappa])

        history = eegnet.fit(X_train, y_train, batch_size=40, epochs=500, validation_data=(X_test, y_test), verbose=2, callbacks=my_callbacks)

        sum_train_acc.append(np.max(history.history['val_binary_accuracy']))
        sum_train_kappa.append(np.max(history.history['val_cohen_kappa']))

        score = eegnet.evaluate(X_eval, y_eval, batch_size=40)
        sum_test_acc.append(score[1])
        sum_test_kappa.append(score[2])
        print(score)


        train_acc = []
        train_kappa = []
        test_acc = []
        test_kappa = []

        train_acc.append([np.mean(sum_train_acc), np.std(sum_train_acc)])
        train_kappa.append([np.mean(sum_train_kappa), np.std(sum_train_kappa)])
        test_acc.append([np.mean(sum_test_acc), np.std(sum_test_acc)])
        test_kappa.append([np.mean(sum_test_kappa), np.std(sum_test_kappa)])

    print('Train Accuracy --->', train_acc)
    print('Train Kappa --->', train_kappa)
    print('Test Accuracy --->', test_acc)
    print('Test Kappa --->', test_kappa)


#### MAIN
#### Iterate thorugh all participants and generate results for various classifiers
participants = np.arange(1, 10)
for i in participants:
    print('-----Participant A0', i, ' --------')
    X, y = get_train_data(fno=i)
    X_eval, y_eval = get_eval_data(fno=i)
    print('EEGNet')
    network_train(X, y, X_eval, y_eval)