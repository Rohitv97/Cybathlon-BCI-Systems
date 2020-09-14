import numpy as np
import pandas as pd
import os
from scipy.io import loadmat

def load_david_data(mode):
    """
    Loading Cybathlon Pilot Data
    :params: mode = Determines which session to load from Cybathlon Pilot Data
    :return: Dataframe containing the EEG data in the shape (samples, channels)

    """
    if mode == 'D1S1':
        path = "data/david_20191008_session"
        f_ext = ".mat"
        mode_path = "1_"
        data = pd.DataFrame()
        for i in range(1, 11):
            value = str(i).zfill(2)
            load_path = path + mode_path + value + f_ext
            file_data = loadmat(load_path)
            mdata = file_data['data']
            temp = pd.DataFrame(mdata)
            data = pd.concat([data, temp])

        data = data.drop([64, 65, 66, 67, 68, 69, 70, 71], axis=1)
        return data
    elif mode=='D1S2':
        path = "data/david_20191008_session"
        f_ext = ".mat"
        mode_path = "2_"
        data = pd.DataFrame()
        for i in range(1, 11):
            value = str(i).zfill(2)
            load_path = path + mode_path + value + f_ext
            file_data = loadmat(load_path)
            mdata = file_data['data']
            temp = pd.DataFrame(mdata)
            data = pd.concat([data, temp])
        
        data = data.drop([64, 65, 66, 67, 68, 69, 70, 71], axis=1)
        return data
    elif mode=='D2S1':
        path = "data/david_20191107_session"
        f_ext = ".npy"
        mode_path = "1_"
        data = pd.DataFrame()
        for i in range(1, 11):
            value = str(i).zfill(2)
            load_path = path + mode_path + value + f_ext
            temp = pd.DataFrame(np.load(load_path))
            data = pd.concat([data, temp])

        data = data.drop([64, 65], axis=1)
        return data
    elif mode=='D2S2':
        path = "data/david_20191107_session"
        f_ext = ".npy"
        mode_path = "2_"
        data = pd.DataFrame()
        for i in range(1, 11):
            value = str(i).zfill(2)
            load_path = path + mode_path + value + f_ext
            temp = pd.DataFrame(np.load(load_path))
            data = pd.concat([data, temp])

        data = data.drop([64, 65], axis=1)
        return data
    elif mode=='D3S1':
        f_ext = ".mat"
        path = "data/david_20200123_session"
        mode_path = "1_"
        data = pd.DataFrame()
        for i in range(1, 9):
            value = str(i).zfill(2)
            load_path = path + mode_path + value + f_ext
            file_data = loadmat(load_path)
            mdata = file_data['data']
            temp = pd.DataFrame(mdata)
            data = pd.concat([data, temp])

        data = data.drop([64, 65, 66, 67, 68, 69, 70, 71], axis=1)
        return data
    elif mode=='D4S1':
        f_ext = ".mat"
        mode_path = "1"
        data = pd.DataFrame()
        path = "data/david_20200206_session"
        load_path = path + mode_path + f_ext
        file_data = loadmat(load_path)
        mdata = file_data['data']
        temp = pd.DataFrame(mdata)
        data = pd.concat([data, temp])
        data = data.drop([64, 65, 66, 67, 68, 69, 70, 71], axis=1)
        return data
    elif mode=='D5S1':
        f_ext = ".mat"
        mode_path = "1"
        data = pd.DataFrame()
        path = "data/david_20200220_session"
        load_path = path + mode_path + f_ext
        file_data = loadmat(load_path)
        mdata = file_data['data']
        temp = pd.DataFrame(mdata)
        data = pd.concat([data, temp])
        data = data.drop([64, 65, 66, 67, 68, 69, 70, 71], axis=1)
        return data


def drop_classes(df, mode):
    """
    Dropping additional classes which are not required
    :params: df = The dataframe with the EEG data
    :params: mode = Determines which session to load from Cybathlon Pilot Data
    :return: Dataframe containing the EEG data in the shape (samples, channels).
    
    """
    if mode in ['D1S1', 'D1S2']:
        label_not_inc = list(range(2,9))
        indexes_to_drop = []
        i = 0
        while i < len(df):
            if df[72].values[i] in label_not_inc:
                list2 = list(range(i, 2048+i))
                i += 2048
                indexes_to_drop.extend(list2)
            else:
                i += 1
        indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
        df_sliced = df.take(list(indexes_to_keep))

        df_sliced = df_sliced.reset_index(drop=True)
        return df_sliced

    elif mode in ['D2S1', 'D2S2']:

        label_not_inc = list(range(2,9))
        indexes_to_drop = []
        i = 0
        while i < len(df):
            if df[66].values[i] in label_not_inc:
                list2 = list(range(i, 767+i))
                i += 767
                indexes_to_drop.extend(list2)
            else:
                i += 1
        indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
        df_sliced = df.take(list(indexes_to_keep))

        df_sliced = df_sliced.reset_index(drop=True)
        return df_sliced

    elif mode in ['D3S1', 'D4S1', 'D5S1']:
        print('Nothing to drop')
        return df

def uniform_label(df, mode):
    """
    Modifying EEG data, so that there are only Hands and Relax class. As such, all Short Hands, Medium Hands,
    and Long hands will just be denoted as Hands.
    :params: df = Dataframe with EEG data
    :params: mode = Determines which session to load from Cybathlon Pilot Data
    :return: dataframe containing the EEG data in the shape (samples, channels) with only 2 classes, hands and relax
    
    """
    if mode not in ['D3S1', 'D4S1', 'D5S1']:
        col = df.columns.values[-1]
        to_keep = [0, 1]
        un = np.unique(df[col].values)
        to_replace = []
        for i in un:
            if i not in to_keep:
                to_replace.append(i)
        print(to_replace)
        df[col] = df[col].replace(to_replace = to_replace, value=2)

        return df
    else:
        print('No changes required. Will be handled by Asynch Label Generator\n')
        return df

def asynch_label_gen(sfreq, df, mode):
    """
    Generate asynchronous (continuous) labels based on trigger points of class
    :params: sfreq = Sampling Frequency of Data
    :params: df = Dataframe with EEG Data
    :params: mode = Determines which session to load from Cybathlon Pilot Data
    :return: Asynchronous labels as numpy array
    
    """
    if mode in ['D1S1', 'D1S2']:
        trigger_points = {1:4, 2:4}
        asynch_label = []
        i = 0
        while i < len(df[72]):
            if(df[72].values[i] == 1):
                end = (sfreq * trigger_points[1]) + 1
                for j in range(i, i+end):
                    asynch_label.append(1)
                i += end
            elif(df[72].values[i] == 2):
                end = (sfreq * trigger_points[2]) + 1
                for j in range(i, i+end):
                    asynch_label.append(2)
                i += end
            elif(df[72].values[i] == 0):
                asynch_label.append(0)
                i += 1

        return np.array(asynch_label)

    elif mode in ['D2S1', 'D2S2']:
        return df.iloc[:, -1].values

    elif mode in ['D3S1', 'D4S1', 'D5S1']:
        trigger_points = {1:6, 2:4, 3:2, 4:2}
        asynch_label = []
        i = 0
        while i < len(df[72]):
            if(df[72].values[i] == 1):
                end = (sfreq * trigger_points[1]) + 1
                for j in range(i, i+end):
                    asynch_label.append(1)
                i += end
            elif(df[72].values[i] == 2):
                end = (sfreq * trigger_points[2]) + 1
                for j in range(i, i+end):
                    asynch_label.append(1)
                i += end
            elif(df[72].values[i] == 3):
                end = (sfreq * trigger_points[3]) + 1
                for j in range(i, i+end):
                    asynch_label.append(1)
                i += end
            elif(df[72].values[i] == 4):
                end = (sfreq * trigger_points[4]) + 1
                for j in range(i, i+end):
                    asynch_label.append(2)
                i += end
            elif(df[72].values[i] == 0):
                asynch_label.append(0)
                i += 1

        return np.array(asynch_label)

        