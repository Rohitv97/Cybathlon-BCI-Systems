import random
import numpy as np
import pandas as pd

def augment_data(df, asynch_label, n_times=5, split_size=2):
    """
    :params: df = EEG data
    :params: asynch_label = Asynchronous labels for EEG data
    :params: n_times = Number of times the process has to be carried out
    :params: split_size = Number of parts to segment an existing EEG trial
    :return: augmented data
    """
    random.seed(42)
    copy_df = df
    copy_df = copy_df.drop(['class'], axis = 1)
    copy_df['class'] = asynch_label

    unique = np.unique(asynch_label)
    flag = 0

    for val in range(len(unique)):
        if unique[val] != 0:
            print('Generating new data for label', unique[val])
            temp_df = copy_df[copy_df['class'] == unique[val]]
            temp_df.reset_index(drop=True)

            divided = []
            i = 0
            while i < 72000:
                temp = temp_df.iloc[i:i+1000, :].values
                divided.append(temp)
                i += 1000  

            divided = np.array(divided)

            k = split_size
            num_of_times = n_times
            size = len(divided[0]) // k

            for i in range(num_of_times):
                parts = []
                choices_main = random.sample(range(0, 72), 2)
                parts.append(divided[choices_main[0]][:size, :])
                parts.append(divided[choices_main[0]][size:, :])
                parts.append(divided[choices_main[1]][:size, :])
                parts.append(divided[choices_main[1]][size:, :])

                choices_final = random.sample(range(0, len(parts)), k)
                new_data_1 = np.concatenate((parts[choices_final[0]], parts[choices_final[1]]))
                new_data_2 = np.concatenate((parts[(len(parts) + choices_final[0]) % 4], parts[(len(parts) + choices_final[1]) % 4]))

                if flag == 0:
                    augmented_data = np.concatenate((new_data_1, new_data_2))
                    flag = 1
                elif flag == 1:
                    augmented_data = np.concatenate((augmented_data, new_data_1, new_data_2))

    return augmented_data