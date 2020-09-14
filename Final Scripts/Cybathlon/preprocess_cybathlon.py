import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP

def band_pass_filter(eeg, freq_range, learning_method):
    """
    Band Pass Filters the EEG data
    :params: eeg = The EEG data
    :params: freq_range = The frequency range between which data has to be bandpassed
    :params: learning_method = Specified which learning method is being used. Different filters are used for different methods.
    :return: Band-passed filtered data
    """
    if learning_method == 'NN':
        info = mne.create_info(64, 512, ch_types=["eeg"] * 64)
        raw = mne.io.RawArray(eeg.T, info)
        raw.filter(freq_range[0], freq_range[1], fir_design='firwin')

        return raw._data.T
    elif learning_method == 'standard':
        info = mne.create_info(64, 512, ch_types=["eeg"] * 64)
        raw = mne.io.RawArray(eeg.T, info)
        iir_params = dict(order=5, ftype='cheby2', rs=2.)
        raw = raw.filter(freq_range[0], freq_range[1], fir_design='firwin', method='iir', iir_params=iir_params)

        return raw._data.T

def data_win(sfreq, data, asynch_label):
    """
    Generating small windows of asynchronous data
    :params: sfreq = Sampling Frequency
    :params: data = EEG data
    :params: asynch_label = The asynchronous labels for the data
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
        if count1 >= 512:
            to_add = 1
        elif count2 >= 512:
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


def preprocess_david(data, asynch_label, learning_method, train=True, csp_objects=None):
    """
    Preprocessing block, calling all necessary functions defined
    :params: data = EEG data
    :params: asynch_label = The asynchronous labels for the data
    :params: learning_method = Learning method used (Either NN or standard)
    :params: train = Boolean to denote whether the data is part of train or test set
    :params: csp_objects = If data is part of train set, this is None. Else, this will contain the CSP objects calculated during training
    :return: Windowed data whose shape is 3D (batch_size, samples, channels)
    :return: CSP Objects (if required)
    :return: final preprocessed data needed to train model
    :return: final preprocessed labels 
    """
    sfreq = 512

    if learning_method == 'NN':
        freq_range = [8, 32]
        out_data = band_pass_filter(data.iloc[:, :-1].values, freq_range=freq_range, learning_method=learning_method)

        X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
        y = transform_label(y)
        X, y = prune_records(X, y)

        dim1, dim2, dim3 = X.shape
        X_new = X.reshape((dim1, 1, dim2, dim3))
        y = y-1

        return X_new, y
    
    elif learning_method == 'standard':
        freq = 4
        increment = 4
        end_freq = 40
        if train==True:
            csp_objects = []
            csp_data = []
            while freq < end_freq:
                freq_range = []
                freq_range.append(freq)
                freq_range.append(freq+increment)
                freq += increment
                out_data = band_pass_filter(data.iloc[:, :-1].values, freq_range=freq_range, learning_method=learning_method)

                X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
                y = transform_label(y)
                X, y = prune_records(X, y)

                csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
                final_data = csp.fit_transform(X, y)

                csp_objects.append(csp)
                csp_data.append(final_data)
                
            csp_data = np.array(csp_data)

            final_data = pd.DataFrame(csp_data[0])
            col_count = 4
            for i in range(1, len(csp_data)):
                for j in range(len(csp_data[i].T)):
                    final_data[str(col_count)] = csp_data[i].T[j]
                    col_count += 1

            return np.array(csp_objects), final_data.values, y

        else:
            csp_data = []
            count = 0
            while freq < end_freq:
                freq_range = []
                freq_range.append(freq)
                freq_range.append(freq+increment)
                freq += increment
                out_data = band_pass_filter(data.iloc[:, :-1].values, freq_range=freq_range, learning_method=learning_method)

                X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=asynch_label)
                y = transform_label(y)
                X, y = prune_records(X, y)

                final_data = csp_objects[count].transform(X)
                count += 1

                csp_data.append(final_data)

            csp_data = np.array(csp_data)

            final_data = pd.DataFrame(csp_data[0])
            col_count = 4
            for i in range(1, len(csp_data)):
                for j in range(len(csp_data[i].T)):
                    final_data[str(col_count)] = csp_data[i].T[j]
                    col_count += 1

            return final_data.values, y
        