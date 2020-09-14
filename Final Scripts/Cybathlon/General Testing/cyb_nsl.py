import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, models
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import CohenKappa
from tensorflow.keras.constraints import max_norm
import neural_structured_learning as nsl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import random

from load_cybathlon import load_david_data, drop_classes, uniform_label, asynch_label_gen
from preprocess_cybathlon import preprocess_david
from data_augment_cybathlon import augment_data

def EEGNET(channels=64, samples=1024):
    """
    Define EEGNet architecture
    """
    input1 = layers.Input(shape=(1, channels, samples), name='input')
    b1 = layers.Conv2D(8, (1, 512), padding='same', use_bias=False, data_format='channels_first')(input1)
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

def network_train(X_train, X_test, y_train, y_test, X_eval, y_eval, X_eval2, y_eval2, mode='normal'):
    """
    Training out model
    :params: X_train, X_test = Attributes of Train and Test sets
    :params: y_train, y_test = Labels of Train and Test sets
    :params: X_eval, y_eval = Arrtibutes and Labels of the evaluation set
    """
    if mode == 'normal':
        sum_train_acc = []
        sum_test_acc = []
        sum_train_kappa = []
        sum_test_kappa = []

        times = 50

        ba = BinaryAccuracy()
        kappa = CohenKappa(num_classes=2)

        #### Train model 50 times and take average results
        for i in range(times):
            eegnet = EEGNET(channels=64, samples=1024)

            adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.001)
            adv_model = nsl.keras.AdversarialRegularization(eegnet, adv_config=adv_config)
            
            my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
            ]

            adv_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=[ba, kappa])

            history = adv_model.fit({'input': X_train, 'label': y_train}, batch_size=25, epochs=500, shuffle=True,
                                validation_data={'input': X_test, 'label': y_test},
                                verbose=2, callbacks=my_callbacks)

            sum_train_acc.append(np.max(history.history['val_binary_accuracy']))
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


            acc = history.history['binary_accuracy']
            val_acc = history.history['val_binary_accuracy']

        print('Train Accuracy --->', train_acc)
        print('Train Kappa --->', train_kappa)
        print('Test Accuracy 1--->', test_acc)
        print('Test Kappa 1--->', test_kappa)

    elif mode == 'final':
        sum_train_acc = []
        sum_test_acc1 = []
        sum_test_acc2 = []
        sum_train_kappa = []
        sum_test_kappa1 = []
        sum_test_kappa2 = []

        times = 50

        for i in range(times):
        eegnet = EEGNET(channels=64, samples=1024)

        adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.001)
        adv_model = nsl.keras.AdversarialRegularization(eegnet, adv_config=adv_config)

        my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        ]

        adv_model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=[ba, kappa])

        history = adv_model.fit({'input': X_train, 'label': y_train}, batch_size=25, epochs=500, shuffle=True,
                                validation_data={'input': X_test, 'label': y_test},
                                verbose=2, callbacks=my_callbacks)


        print('-----')
        print(history.history['val_binary_accuracy'])
        print(history.history['val_cohen_kappa'])
        sum_train_acc.append(np.max(history.history['val_binary_accuracy']))
        sum_train_kappa.append(np.max(history.history['val_cohen_kappa']))

        score = adv_model.evaluate({'input': X_eval1, 'label': y_eval1})
        sum_test_acc1.append(score[2])
        sum_test_kappa1.append(score[3])
        print(score)

        score = adv_model.evaluate({'input': X_eval2, 'label': y_eval2})
        sum_test_acc2.append(score[2])
        sum_test_kappa2.append(score[3])
        print(score)


        train_acc = []
        train_kappa = []
        test_acc1 = []
        test_acc2 = []
        test_kappa1 = []
        test_kappa2 = []

        train_acc.append([np.mean(sum_train_acc), np.std(sum_train_acc)])
        train_kappa.append([np.mean(sum_train_kappa), np.std(sum_train_kappa)])
        test_acc1.append([np.mean(sum_test_acc1), np.std(sum_test_acc1)])
        test_kappa1.append([np.mean(sum_test_kappa1), np.std(sum_test_kappa1)])
        test_acc2.append([np.mean(sum_test_acc2), np.std(sum_test_acc2)])
        test_kappa2.append([np.mean(sum_test_kappa2), np.std(sum_test_kappa2)])

        print('Train Accuracy --->', train_acc)
        print('Train Kappa --->', train_kappa)
        print('Test Accuracy 1--->', test_acc1)
        print('Test Kappa 1--->', test_kappa1)
        print('Test Accuracy 2--->', test_acc2)
        print('Test Kappa 2--->', test_kappa2)


to_augment = True #False
#False if we don't need augmented data

### Listing all sessions of data
sessions = {0: 'D1S1', 1: 'D1S2', 2: 'D2S1', 3:'D2S2',
            4: 'D3S1', 5: 'D4S1', 6: 'D5S1'}

sfreq = 512 #Sampling frequency
tf.keras.backend.clear_session()

### Test Case 1 ###
print('---Test Case 1-----D1S1 and D1S2------')
## train df
train = sessions[0]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

train_df = df

## eval df
train = sessions[1]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

eval_df = df

asynch_label = train_df.iloc[:, -1].values

if to_augment == True:
    augmented_data = augment_data(train_df, asynch_label, n_times=250, split_size=2)
    extra_label = augmented_data[:, -1]

    augmented_data = pd.DataFrame(augmented_data)
    augmented_data = augmented_data.rename(columns = {64: 'asynch'})
    train_df = pd.concat([train_df, augmented_data])

    asynch_label = np.concatenate((asynch_label, extra_label))

### Preprocess data and train models
X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='NN')
X_eval, y_eval = preprocess_david(eval_df, eval_df.iloc[:, -1].values, learning_method='NN')

X = X_full
y = y_full
print(X.shape)
dim1, dim2, dim3, dim4 = X.shape
split_idx = ((75 * dim1) // 100) + 1
X_train = X[:split_idx, :, :, :]
y_train = y[:split_idx]
X_test = X[split_idx:, :, :, :]
y_test = y[split_idx:]
# X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

network_train(X_train, X_test, y_train, y_test, X_eval, y_eval)

### Test Case 2 ###
print('---Test Case 2-----D2S1 and D2S2------')
## train df
train = sessions[2]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

train_df = df

## eval df
train = sessions[3]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

eval_df = df

asynch_label = train_df.iloc[:, -1].values

if to_augment == True:
    augmented_data = augment_data(train_df, asynch_label, n_times=250, split_size=2)
    extra_label = augmented_data[:, -1]

    augmented_data = pd.DataFrame(augmented_data)
    augmented_data = augmented_data.rename(columns = {64: 'asynch'})
    train_df = pd.concat([train_df, augmented_data])

    asynch_label = np.concatenate((asynch_label, extra_label))

### Preprocess data and train models
X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='NN')
X_eval, y_eval = preprocess_david(eval_df, eval_df.iloc[:, -1].values, learning_method='NN')

X = X_full
y = y_full
print(X.shape)
dim1, dim2, dim3, dim4 = X.shape
split_idx = ((75 * dim1) // 100) + 1
X_train = X[:split_idx, :, :, :]
y_train = y[:split_idx]
X_test = X[split_idx:, :, :, :]
y_test = y[split_idx:]
# X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

network_train(X_train, X_test, y_train, y_test, X_eval, y_eval)

### Test Case 3 ###
print('---Test Case 3-----D3S1 and D4S1 and D5S1------')
## train df
train = sessions[4]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

train_df = df

## eval1 df
train = sessions[5]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

eval_df1 = df

## eval2 df
train = sessions[6]
df = load_david_data(train)
df = drop_classes(df, train)
df = uniform_label(df, train)
asynch_label = asynch_label_gen(sfreq, df, train)
col = df.columns[-1]
df = df.drop(col, axis=1)
df['asynch'] = asynch_label

eval_df2 = df

asynch_label = train_df.iloc[:, -1].values

if to_augment == True:
    augmented_data = augment_data(train_df, asynch_label, n_times=250, split_size=2)
    extra_label = augmented_data[:, -1]

    augmented_data = pd.DataFrame(augmented_data)
    augmented_data = augmented_data.rename(columns = {64: 'asynch'})
    train_df = pd.concat([train_df, augmented_data])

    asynch_label = np.concatenate((asynch_label, extra_label))

### Preprocess data and train models
X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='NN')
X_eval1, y_eval1 = preprocess_david(eval_df1, eval_df1.iloc[:, -1].values, learning_method='NN')
X_eval2, y_eval2 = preprocess_david(eval_df2, eval_df2.iloc[:, -1].values, learning_method='NN')

X = X_full
y = y_full
print(X.shape)
dim1, dim2, dim3, dim4 = X.shape
split_idx = ((75 * dim1) // 100) + 1
X_train = X[:split_idx, :, :, :]
y_train = y[:split_idx]
X_test = X[split_idx:, :, :, :]
y_test = y[split_idx:]
# X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.25, random_state=42)

network_train(X_train, X_test, y_train, y_test, X_eval1, y_eval1, X_eval2, y_eval2, mode='final')