import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import random
import sys

from graz_augment import augment_data

def band_pass_filter(eeg, freq_range):
    """
    Band Pass Filtering
    :params: eeg = EEG data
    :params: freq_range = Frequency range for band-pass filtering
    :return: Filtered data
    """
    sfreq = 250 #sampling frequency
    info = mne.create_info(22, sfreq, ch_types=["eeg"] * 22)
    raw = mne.io.RawArray(eeg.T, info)
    iir_params = dict(order=5, ftype='cheby2', rs=2.)
    raw = raw.filter(freq_range[0], freq_range[1], fir_design='firwin', method='iir', iir_params=iir_params)

    return raw._data.T

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

### Alternate method for Transform Label. (performs poorer in general)
# def transform_label(label_new):
#   label = []

#   for i in label_new:
#     to_add = i[-1]
#     label.append(to_add)

#   return np.array(label)

def fbcsp(df, labels, sfreq, train=True, csp_objects=None):
    """
    Implements the Filter Bank CSP
    :params: df = EEG data
    :params: labels = Class labels for the data
    :params: sfreq = Sampling Frequency
    :params: train = Bool. If True, then CSP objects are fitted on data. If False, then previously fitted objects are used
    :params: csp_objects = Previously fitted CSP objects
    :return: csp_objects which are fitted on data
    :return: Output of FBCSP algorithm
    :return: processed class labels for the output
    """
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
            out_data = band_pass_filter(df.iloc[:, :-1].values, freq_range=freq_range)

            X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=labels)
            y = transform_label(y)
            X, y = prune_records(X, y)

            csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
            final_data = csp.fit_transform(X, y)

            csp_objects.append(csp)
            csp_data.append(final_data)

        return np.array(csp_objects), np.array(csp_data), y

    else:
        csp_data = []
        count = 0
        while freq < end_freq:
            freq_range = []
            freq_range.append(freq)
            freq_range.append(freq+increment)
            freq += increment
            out_data = band_pass_filter(df.iloc[:, :-1].values, freq_range=freq_range)

            X, y = data_win(sfreq=sfreq, data=out_data, asynch_label=labels)
            y = transform_label(y)
            X, y = prune_records(X, y)

            final_data = csp_objects[count].transform(X)
            count += 1

            csp_data.append(final_data)

        return np.array(csp_data), y

def itr(n_class, p_class, c_time):
    """
    Implementation of Information Transfer Rate
    :params: n_class = number of classes
    :params: p_class = probability (accuracy)
    :params: c_time = time taken to classify
    :return: ITR
    """
    B = (np.log2(n_class) + (p_class * np.log2(p_class)) + ((1-p_class) * np.log2((1-p_class)/(n_class-1)))) / c_time * 60

    return B

def performance_metrics(y_test, y_pred):
    """
    Calculates verious performance metrics
    :params: y_test = Ground truth
    :params: y_pred = Predicted labels
    """
    acc = accuracy_score(y_test, y_pred)
    print('Accuracy Score: ', acc)
    print('Cohen Kappa Score: ', cohen_kappa_score(y_test, y_pred))
    print('ITR (bits per minute): ', itr(n_class=2, p_class=acc, c_time=2))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))

def get_train_data(fno):
    """
    Function to load and preprocess the training data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}

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

    csp_objects, csp_data, y = fbcsp(train_df, asynch_label, sfreq, train=True)

    final_data = pd.DataFrame(csp_data[0])
    col_count = 4
    for i in range(1, len(csp_data)):
        for j in range(len(csp_data[i].T)):
            final_data[str(col_count)] = csp_data[i].T[j]
            col_count += 1

    return final_data.values, y, csp_objects

def get_eval_data(fno, csp_objects):
    """
    Function to load and preprocess the testing data for a volunteer
    :params: fnp = Participant number
    :return: EEG data
    """
    sfreq = 250 #Sampling frequency
    trigger_points = {1:4, 2:4, 3:4, 4:4}

    df = load_data(mode='test', fno=fno)
    train_df = drop_classes(df)
    train_df = train_df.drop([22, 23, 24], axis=1)
    asynch_label = convert_class(sfreq=sfreq, trigger_points=trigger_points, df=train_df)
    csp_data, y = fbcsp(train_df, asynch_label, sfreq, train=False, csp_objects=csp_objects)

    final_data = pd.DataFrame(csp_data[0])
    col_count = 4
    for i in range(1, len(csp_data)):
        for j in range(len(csp_data[i].T)):
            final_data[str(col_count)] = csp_data[i].T[j]
            col_count += 1

    return final_data.values, y

def train_mibif(X, y):
    """
    Trains FBCSP + MIBIF Classifiers
    :params: X = training features
    :params: y = Training labels
    :return: positions of Most informative or selected features
    :return: trained SVM and LDA classifier
    :return: Fitted standard scaler object
    """
    print('------Train---------')
    dim1, dim2 = X.shape
    split_idx = ((75 * dim1) // 100) + 1
    X_train = X[:split_idx, :]
    y_train = y[:split_idx]
    X_test = X[split_idx:, :]
    y_test = y[split_idx:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    #get the best k features base on MIBIF algorithm
    select_K = SelectKBest(mutual_info_classif,k=8).fit(X, y)
    extra = select_K.get_support()
    if extra[0] == True:
        extra[1] = True
    for i in range(2, len(extra)):
        if extra[i] == True:
            if i%2 == 0:
                extra[i+1] = True
            else:
                extra[i-1] = True
    pos = np.where(extra==False)
    New_train = np.delete(X_train, list(pos[0]), 1)
    New_test = np.delete(X_test, list(pos[0]), 1)
    ss = StandardScaler()
    New_train = ss.fit_transform(New_train,y_train)
    New_test = ss.transform(New_test)

    print('####### SVM#####')
    svm = SVC()
    svm.fit(New_train, y_train)
    y_pred = svm.predict(New_test)
    performance_metrics(y_test, y_pred)
    
    print('##########LDA#########')
    lda = LinearDiscriminantAnalysis()
    lda.fit(New_train, y_train)
    y_pred = lda.predict(New_test)
    performance_metrics(y_test, y_pred)

    return list(pos[0]), svm, lda, ss

def eval_mibif(pos, svm, lda, ss, X, y):
    """
    Evaluation of FBCSP + MIBIF classifiers
    :param: pos = indicates positions of informative and non-informative features
    :param: svm = Train SVM classifier
    :param: lda = Trained LDA classifier
    :param: ss = Fitted Standard scaler object
    :param: X = Features or attributes
    :param: y = Ground truth labels
    """
    print('------Test--------')
    X = np.delete(X, pos, 1)
    X = ss.transform(X)
    print('#####SVM######')
    y_pred = svm.predict(X)
    performance_metrics(y, y_pred)
    print('#####LDA######')
    y_pred = lda.predict(X)
    performance_metrics(y, y_pred)

def train_rf(X, y):
    """
    Trains FBCSP + Random Forest Classifier
    :params: X = training features
    :params: y = Training labels
    :return: trained RF classifier
    :return: Fitted standard scaler object
    """
    print('--------Train--------')
    dim1, dim2 = X.shape
    split_idx = ((75 * dim1) // 100) + 1
    X_train = X[:split_idx, :]
    y_train = y[:split_idx]
    X_test = X[split_idx:, :]
    y_test = y[split_idx:]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train, y_train)
    clf.fit(X_train, y_train)
    X_test = ss.transform(X_test)
    y_pred = clf.predict(X_test)
    ####Random Forest#######
    performance_metrics(y_test, y_pred)

    return clf, ss

def eval_rf(clf, ss, X, y):
    """
    Evaluation of FBCSP + RF classifier
    :param: clf = Trained RF classifier
    :param: ss = Fitted Standard scaler object
    :param: X = Features or attributes
    :param: y = Ground truth labels
    """
    print('--------Test------')
    X = ss.transform(X)
    y_pred = clf.predict(X)
    performance_metrics(y, y_pred)

#### MAIN
#### Iterate thorugh all participants and generate results for various classifiers
participants = np.arange(1, 10)
for i in participants:
    print('-----Participant A0', i, ' --------')
    X, y, csp_objects = get_train_data(fno=i)
    X_eval, y_eval = get_eval_data(fno=i,csp_objects=csp_objects)
    print('FBCSP + MIBIF (SVM and LDA)')
    pos, svm, lda, ss = train_mibif(X, y)
    eval_mibif(pos, svm, lda, ss, X_eval, y_eval)
    print('FBCSP + RF')
    clf, ss = train_rf(X, y)
    eval_rf(clf, ss, X_eval, y_eval)