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
    sfreq = 200
    info = mne.create_info(22, 200, ch_types=["eeg"] * 22)
    raw = mne.io.RawArray(eeg.T, info)
    iir_params = dict(order=11, ftype='cheby2', rs=2.)
    raw = raw.filter(freq_range[0], freq_range[1], fir_design='firwin', method='iir', iir_params=iir_params)

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


def get_data(subject='B'):
    """
    Function to get all data of a particular participant
    :params: subject = Specifies the participant
    :return: Data from 3 different sessions.
    """
    sfreq = 200 #Sampling frequency
    trigger_points = {1:4, 2:4}

    to_augment = True #False
    #True if you want to augment the dataset, else set it to False

    train_df, eval1_df, eval2_df = load_data(subject)
    train_df = drop_classes(train_df)
    eval1_df = drop_classes(eval1_df)
    eval2_df = drop_classes(eval2_df)

    asynch_label = train_df.iloc[:, -1].values

    if to_augment == True:
        augmented_data = augment_data(train_df, asynch_label, n_times=50, split_size=2)
        extra_label = augmented_data[:, -1]
        augmented_data = pd.DataFrame(augmented_data)
        augmented_data = augmented_data.rename(columns = {22: 'class'})
        train_df = pd.concat([train_df, augmented_data])
        asynch_label = np.concatenate((asynch_label, extra_label))

    ### FBCSP for train data
    csp_objects, csp_data, y_train = fbcsp(train_df, asynch_label, sfreq, train=True)
    final_data = pd.DataFrame(csp_data[0])
    col_count = 4
    for i in range(1, len(csp_data)):
        for j in range(len(csp_data[i].T)):
            final_data[str(col_count)] = csp_data[i].T[j]
            col_count += 1
    x_train = final_data.values

    ### FBCSP for evaluation data 1
    csp_data, y_eval1 = fbcsp(eval1_df, eval1_df.iloc[:, -1].values, sfreq, train=False, csp_objects=csp_objects)
    final_data = pd.DataFrame(csp_data[0])
    col_count = 4
    for i in range(1, len(csp_data)):
        for j in range(len(csp_data[i].T)):
            final_data[str(col_count)] = csp_data[i].T[j]
            col_count += 1
    x_eval1 = final_data.values

    ### FBCSP for evaluation data 2
    csp_data, y_eval2 = fbcsp(eval2_df, eval2_df.iloc[:, -1].values, sfreq, train=False, csp_objects=csp_objects)
    final_data = pd.DataFrame(csp_data[0])
    col_count = 4
    for i in range(1, len(csp_data)):
        for j in range(len(csp_data[i].T)):
            final_data[str(col_count)] = csp_data[i].T[j]
            col_count += 1
    x_eval2 = final_data.values

    return x_train, y_train, x_eval1, y_eval1, x_eval2, y_eval2


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
    select_K = SelectKBest(mutual_info_classif,k=10).fit(X, y)
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

participants = ['B', 'C', 'E', 'F']

for i in range(len(participants)):
    sub = participants[i]
    print('----Participant ', sub, ' ------')
    ###SVM and LDA
    x_train, y_train, x_eval1, y_eval1, x_eval2, y_eval2 = get_data(sub)
    pos, svm, lda, ss = train_mibif(x_train, y_train)
    print('------Eval1--------')
    eval_mibif(pos, svm, lda, ss, x_eval1, y_eval1)
    print('------Eval2--------')
    eval_mibif(pos, svm, lda, ss, x_eval2, y_eval2)

    ### Random Forest
    clf, ss = train_rf(x_train, y_train)
    print('------eval1--------')
    eval_rf(clf, ss, x_eval1, y_eval1)
    print('------eval2--------')
    eval_rf(clf, ss, x_eval2, y_eval2)