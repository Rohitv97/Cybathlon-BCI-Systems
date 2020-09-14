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
import neural_structured_learning as nsl
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, RMSprop
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
    iir_params = dict(order=4, ftype='butter')
    raw.filter(freq_range[0], freq_range[1], fir_design='firwin', method='iir', iir_params=iir_params)

    return raw._data.T

def get_train_data(fno):
    """
    Function to load and preprocess the training data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}
    freq_range = [0.5, 100]

    df = load_data(mode='train', fno=fno)
    df = df.drop([22, 23, 24], axis=1)
    out_data = band_pass_filter(df.iloc[:, :-1].values, freq_range=freq_range)

    df_new = pd.DataFrame(out_data)
    df_new['label'] = df.iloc[:, -1].values

    info = mne.create_info(23, sfreq, ch_types=["eeg"] * 22 + ['stim'] * 1)
    raw = mne.io.RawArray(df_new.T, info)
    events = mne.find_events(raw, stim_channel='22')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    event_dict = {'Left Hands': 1, 'Right Hands': 2, 'Foot': 3, 'Tongue': 4}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-1, tmax=4, picks=picks, preload=True)
    epochs = epochs.crop(tmin=0.5, tmax=3.5)

    y = epochs.events[:, -1]
    X = epochs.get_data()

    dim1, dim2, dim3 = X.shape
    X = X.reshape((dim1, 1, dim2, dim3))

    y = y - 1

    X_train = X[:241, :, :, :]
    X_test = X[241:, :, :, :]
    y_train = y[:241]
    y_test = y[241:]

    return X_train, X_test, y_train, y_test

def get_eval_data(fno):
    """
    Function to load and preprocess the testing data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}
    freq_range = [0.5, 100]

    df = load_data(mode='test', fno=fno)
    df = df.drop([22, 23, 24], axis=1)
    out_data = band_pass_filter(df.iloc[:, :-1].values, freq_range=freq_range)

    df_new = pd.DataFrame(out_data)
    df_new['label'] = df.iloc[:, -1].values

    info = mne.create_info(23, sfreq, ch_types=["eeg"] * 22 + ['stim'] * 1)
    raw = mne.io.RawArray(df_new.T, info)
    events = mne.find_events(raw, stim_channel='22')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                    exclude='bads')
    event_dict = {'Left Hands': 1, 'Right Hands': 2, 'Foot': 3, 'Tongue': 4}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-1, tmax=4, picks=picks, preload=True)
    epochs = epochs.crop(tmin=0.5, tmax=3.5)

    y = epochs.events[:, -1]
    X = epochs.get_data()

    dim1, dim2, dim3 = X.shape
    X = X.reshape((dim1, 1, dim2, dim3))

    y = y - 1

    return X, y

def EEGNET(channels=22, samples=500):
    """
    Implementing EEGNet Architecture
    """
    input1 = layers.Input(shape=(1, channels, samples), name='input')
    b1 = layers.Conv2D(8, (1, 30), padding='same', use_bias=False, data_format='channels_first')(input1)
    b1 = layers.BatchNormalization(axis=1)(b1)
    b1 = layers.DepthwiseConv2D((channels, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.), data_format='channels_first')(b1)
    b1 = layers.BatchNormalization(axis=1)(b1)
    b1 = layers.Activation('elu')(b1)
    b1 = layers.AveragePooling2D((1, 4), data_format='channels_first')(b1)
    b1 = layers.Dropout(0.25)(b1)

    b2 = layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False, data_format='channels_first')(b1)
    b2 = layers.BatchNormalization(axis=1)(b2)
    b2 = layers.Activation('elu')(b2)
    b2 = layers.AveragePooling2D((1, 8), data_format='channels_first')(b2)
    b2 = layers.Dropout(0.25)(b2)

    flatten = layers.Flatten()(b2)

    dense = layers.Dense(4, kernel_constraint=max_norm(0.25))(flatten)
    activation = layers.Activation('softmax')(dense)

    eegnet = Model(inputs = input1, outputs=activation)

    return eegnet

def network_train(X_train, y_train, X_test, y_test, X_eval, y_eval):
    """
    Trains the network multiple times and takes average results
    :params: X_train = training features
    :params: y_train = training labels
    :params: X_test = validation features
    :params: y_test = Validation labels
    :params: X_eval = Evaluation features
    :params: y_eval = Evaluation labels
    """
    y_train = to_categorical(y_train, 4)
    y_test = to_categorical(y_test, 4)
    y_eval = to_categorical(y_eval, 4)

    sum_train_acc = []
    sum_test_acc = []
    sum_train_kappa = []
    sum_test_kappa = []

    kappa = CohenKappa(num_classes=4)
    rmsprop = RMSprop(learning_rate=0.001, momentum=0.4)

    times = 25

    dim1, dim2, dim3, dim4 = X_train.shape

    for i in range(times):
        eegnet = EEGNET(channels=22, samples=dim4)

        adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.5,  adv_grad_norm='infinity')
        adv_model = nsl.keras.AdversarialRegularization(eegnet, adv_config=adv_config)

        my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
        ]

        adv_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['acc', kappa])

        history = adv_model.fit({'input': X_train, 'label': y_train}, batch_size=32, epochs=300,
                                validation_data={'input': X_test, 'label': y_test},
                                verbose=2, callbacks=my_callbacks)

        sum_train_acc.append(np.max(history.history['val_categorical_accuracy']))
        sum_train_kappa.append(np.max(history.history['val_cohen_kappa']))

        score = adv_model.evaluate({'input': X_eval, 'label': y_eval})
        sum_test_acc.append(score[2])
        sum_test_kappa.append(score[3])
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

    return history


#### MAIN
#### Iterate thorugh all participants and generate results for various classifiers
participants = np.arange(1, 10)
for i in participants:
    print('-----Participant A0', i, ' --------')
    X_train, X_test, y_train, y_test = get_train_data(fno=i)
    X_eval, y_eval = get_eval_data(fno=i)
    print('NSL')
    network_train(X_train, y_train, X_test, y_test, X_eval, y_eval)