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
from kaya_augment import augment_data

def load_data(subject='B'):
    """
    Loading Data
    :params: subject = Subject whose data has to be loaded
    :return: Return the training data, as well as validation datasets
    """
    if subject=='B':
        train_path = "/content/drive/My Drive/BCI/CLASubjectB1510193StLRHand.mat"
        eval1_path = "/content/drive/My Drive/BCI/CLASubjectB1510203StLRHand.mat"
        eval2_path = "/content/drive/My Drive/BCI/CLASubjectB1512153StLRHand.mat"        
    elif subject=='C':
        train_path = "/content/drive/My Drive/BCI/CLASubjectC1511263StLRHand.mat"
        eval1_path = "/content/drive/My Drive/BCI/CLASubjectC1512163StLRHand.mat"
        eval2_path = "/content/drive/My Drive/BCI/CLASubjectC1512233StLRHand.mat" 
    elif subject=='E':
        train_path = "/content/drive/My Drive/BCI/CLASubjectE1512253StLRHand.mat"
        eval1_path = "/content/drive/My Drive/BCI/CLASubjectE1601193StLRHand.mat"
        eval2_path = "/content/drive/My Drive/BCI/CLASubjectE1601223StLRHand.mat" 
    elif subject=='F':
        train_path = "/content/drive/My Drive/BCI/CLASubjectF1509163StLRHand.mat"
        eval1_path = "/content/drive/My Drive/BCI/CLASubjectF1509173StLRHand.mat"
        eval2_path = "/content/drive/My Drive/BCI/CLASubjectF1509283StLRHand.mat" 
    elif subject=='J':
        train_path = "/content/drive/My Drive/BCI/CLA-SubjectJ-170504-3St-LRHand-Inter.mat"
        eval1_path = "/content/drive/My Drive/BCI/CLA-SubjectJ-170508-3St-LRHand-Inter.mat"
        eval2_path = "/content/drive/My Drive/BCI/CLA-SubjectJ-170510-3St-LRHand-Inter.mat"

    file_data = loadmat(train_path)
    data = file_data['o']
    train_df = pd.DataFrame(data[0][0][5])
    train_df['class'] = data[0][0][4]
    
    file_data = loadmat(eval1_path)
    data = file_data['o']
    eval1_df = pd.DataFrame(data[0][0][5])
    eval1_df['class'] = data[0][0][4]
    
    file_data = loadmat(eval2_path)
    data = file_data['o']
    eval2_df = pd.DataFrame(data[0][0][5])
    eval2_df['class'] = data[0][0][4]
    
    return train_df, eval1_df, eval2_df

def band_pass_filter(eeg, freq_range):
    """
    Band Pass Filtering
    :params: eeg = EEG data
    :params: freq_range = Frequency range for band-pass filtering
    :return: Filtered data
    """
    info = mne.create_info(22, 200, ch_types=["eeg"] * 22)
    raw = mne.io.RawArray(eeg.T, info)
    raw.filter(freq_range[0], freq_range[1], fir_design='firwin')

    return raw._data.T

def drop_classes(df):
    """
    Drop un-required classes if any
    :params: df = EEG data
    :return: EEG data after dropping classes
    """
    to_keep = [0,1,2]
    df = df[df['class'].isin([0,1,2])]
    df = df.reset_index(drop=True)

    return df

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
        count9 = np.count_nonzero(i==2)
        if count1 >= 200:
            to_add = 1
        elif count9 >= 200:
            to_add = 2
        else:
            to_add = 0
        label.append(to_add)
        
    label = np.array(label)
    return label

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

def get_data(subject='B'):
    """
    Function to get all data of a particular participant
    :params: subject = Specifies the participant
    :return: Data from 3 different sessions.
    """
    sfreq = 200 #Sampling frequency
    trigger_points = {1:4, 2:4}
    freq_range = [8, 32]
    
    to_augment = True #False
    #True if you want to augment the dataset, else set it to False

    train_df, eval1_df, eval2_df = load_data(subject)
    train_df = drop_classes(train_df)
    eval1_df = drop_classes(eval1_df)
    eval2_df = drop_classes(eval2_df)

    asynch_label = train_df.iloc[:, -1].values

    if to_augment == True:
        augmented_data = augment_data(train_df, asynch_label, n_times=100, split_size=2)
        extra_label = augmented_data[:, -1]
        augmented_data = pd.DataFrame(augmented_data)
        augmented_data = augmented_data.rename(columns = {22: 'class'})
        train_df = pd.concat([train_df, augmented_data])
        asynch_label = np.concatenate((asynch_label, extra_label))

    ### Processing Training Data
    out_data = band_pass_filter(train_df.iloc[:, :-1].values, freq_range=freq_range)

    X_train, y_train = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
    y_train = transform_label(y_train)
    X_train, y_train = prune_records(X_train, y_train)

    dim1, dim2, dim3 = X_train.shape
    X_train = X_train.reshape((dim1, 1, dim2, dim3))
    y_train = y_train - 1

    ### Processing Evaluation data 1
    out_data = band_pass_filter(eval1_df.iloc[:, :-1].values, freq_range=freq_range)

    X_eval1, y_eval1 = data_win(sfreq=sfreq, data=out_data, asynch_label=eval1_df.iloc[:, -1].values)
    y_eval1 = transform_label(y_eval1)
    X_eval1, y_eval1 = prune_records(X_eval1, y_eval1)

    dim1, dim2, dim3 = X_eval1.shape
    X_eval1 = X_eval1.reshape((dim1, 1, dim2, dim3))
    y_eval1 = y_eval1 - 1

    ### Processing Evaluation Data 2
    out_data = band_pass_filter(eval2_df.iloc[:, :-1].values, freq_range=freq_range)

    X_eval2, y_eval2 = data_win(sfreq=sfreq, data=out_data, asynch_label=eval2_df.iloc[:, -1].values)
    y_eval2 = transform_label(y_eval2)
    X_eval2, y_eval2 = prune_records(X_eval2, y_eval2)

    dim1, dim2, dim3 = X_eval2.shape
    X_eval2 = X_eval2.reshape((dim1, 1, dim2, dim3))
    y_eval2 = y_eval2 - 1

    return X_train, y_train, X_eval1, y_eval1, X_eval2, y_eval2

def EEGNET(channels=22, samples=400):
    """
    Implementing EEGNet Architecture
    """
    input1 = layers.Input(shape=(1, channels, samples))
    b1 = layers.Conv2D(8, (1, 200), padding='same', use_bias=False, data_format='channels_first')(input1)
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

def network_train(X, y, X_eval1, y_eval1, X_eval2, y_eval2):
    """
    Trains the network multiple times and takes average results
    :params: X = training features
    :params: y = training labels
    :params: X_eval1 = Evluation 1 data set
    :params: y_eval1 = Evaluation 1 labels
    :params: X_eval2 = Evaluation 2 data set
    :params: y_eval2 = Evaluation 2 labels
    """
    print(X.shape)
    dim1, dim2, dim3, dim4 = X.shape
    split_idx = ((75 * dim1) // 100) + 1
    X_train = X[:split_idx, :, :, :]
    y_train = y[:split_idx]
    X_test = X[split_idx:, :, :, :]
    y_test = y[split_idx:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    sum_train_acc = []
    sum_test_acc1 = []
    sum_test_acc2 = []
    sum_train_kappa = []
    sum_test_kappa1 = []
    sum_test_kappa2 = []

    times = 50

    ba = BinaryAccuracy()
    kappa = CohenKappa(num_classes=2)

    for i in range(times):
        eegnet = EEGNET(channels=22, samples=400)

        eegnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=[ba, kappa])
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        my_callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        ]

        history = eegnet.fit(X_train, y_train, batch_size=25, epochs=500, validation_data=(X_test, y_test), verbose=2, callbacks=my_callbacks)

        sum_train_acc.append(np.max(history.history['val_binary_accuracy']))
        sum_train_kappa.append(np.max(history.history['val_cohen_kappa']))

        score = eegnet.evaluate(X_eval1, y_eval1)
        sum_test_acc1.append(score[1])
        sum_test_kappa1.append(score[2])
        print(score)

        score = eegnet.evaluate(X_eval2, y_eval2)
        sum_test_acc2.append(score[1])
        sum_test_kappa2.append(score[2])
        print(score)

        train_acc = []
        train_kappa = []
        test_acc_1 = []
        test_kappa_1 = []
        test_acc_2 = []
        test_kappa_2 = []

        train_acc.append([np.mean(sum_train_acc), np.std(sum_train_acc)])
        train_kappa.append([np.mean(sum_train_kappa), np.std(sum_train_kappa)])
        test_acc_1.append([np.mean(sum_test_acc1), np.std(sum_test_acc1)])
        test_kappa_1.append([np.mean(sum_test_kappa1), np.std(sum_test_kappa1)])
        test_acc_2.append([np.mean(sum_test_acc2), np.std(sum_test_acc2)])
        test_kappa_2.append([np.mean(sum_test_kappa2), np.std(sum_test_kappa2)])

    print('Train Accuracy --->', train_acc)
    print('Train Kappa --->', train_kappa)
    print('Test Accuracy 1--->', test_acc_1)
    print('Test Kappa 1--->', test_kappa_1)
    print('Test Accuracy 2--->', test_acc_2)
    print('Test Kappa 2--->', test_kappa_2)


participants = ['B', 'C', 'E', 'F']

### Extract data from each participant and train model
for i in range(len(participants)):
    sub = participants[i]
    print('----Participant ', sub, ' ------')
    x_train, y_train, x_eval1, y_eval1, x_eval2, y_eval2 = get_data(sub)
    network_train(x_train, y_train, x_eval1, y_eval1, x_eval2, y_eval2)

