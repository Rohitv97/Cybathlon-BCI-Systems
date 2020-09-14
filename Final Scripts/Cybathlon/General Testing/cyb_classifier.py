import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from load_cybathlon import load_david_data, drop_classes, uniform_label, asynch_label_gen
from preprocess_cybathlon import preprocess_david
from data_augment_cybathlon import augment_data

def itr(n_class, p_class, c_time):
  """
  Calculate Information Transfer Rate
  :params: n_class = Number of classes
  :params: p_class = Probability of class being chosen (accuracy)
  :params: c_time = Time taken to classify one trial
  :return: ITR
  """
  B = (np.log2(n_class) + (p_class * np.log2(p_class)) + ((1-p_class) * np.log2((1-p_class)/(n_class-1)))) / c_time * 60

  return B

def performance_metrics(y_test, y_pred):
  """
  Calculate various performance metrics
  :params: y_test = Ground truth labels
  :params: y_pred = Predicted labels
  """
  result = []
  acc = accuracy_score(y_test, y_pred)
  print('Accuracy Score: ', acc)
  print('Cohen Kappa Score: ', cohen_kappa_score(y_test, y_pred))
  print('ITR (bits per minute): ', itr(n_class=2, p_class=acc, c_time=2))
  print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))


def train_mibif(X, y):
  """
  Train FBCSP + MIBIF Classifier
  :params: X = Attributes
  :params: y = labels/classes
  :return: Selected features and standard scaler object
  :return: SVM and LDA trained classifiers
  """
  print('------Train---------')
  #Split the data
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
  Evaluate FBCSP + MIBIF Classifier
  :params: pos = Selected feautres positions
  :params: svm = trained SVM classifier
  :params: lda = trained LDA classifier
  :params: ss = Fitted Standard Scaler Object
  :params: X = Attributes
  :params: y = labels/classes
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
  Train FBCSP + Random Forest Classifier
  :params: X = Attributes
  :params: y = labels/classes
  :return: Standard scaler object
  :return: RF classifier
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
  Evaluate FBCSP + Random Forest Classifier
  :params: clf = Trained RF classifier
  :params: ss = Fitted Standard scaler object
  :params: X = Attributes
  :params: y = labels/classes
  """
  print('--------Test------')
  X = ss.transform(X)
  y_pred = clf.predict(X)
  performance_metrics(y, y_pred)


to_augment = True #False
#False if we don't need augmented data

### Listing all sessions of data
sessions = {0: 'D1S1', 1: 'D1S2', 2: 'D2S1', 3:'D2S2',
            4: 'D3S1', 5: 'D4S1', 6: 'D5S1'}

sfreq = 512 #Sampling frequency

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
csp_objects, X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='standard', train=True)
X_eval, y_eval = preprocess_david(eval_df, eval_df.iloc[:, -1].values, learning_method='standard', train=False, csp_objects=csp_objects)

print('-----MIBIF----')
pos, svm, lda, ss = train_mibif(X_full, y_full)
eval_mibif(pos, svm, lda, ss, X_eval, y_eval)

print('------RF------')
clf, ss = train_rf(X_full, y_full)
eval_rf(clf, ss, X_eval, y_eval)


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
csp_objects, X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='standard', train=True)
X_eval, y_eval = preprocess_david(eval_df, eval_df.iloc[:, -1].values, learning_method='standard', train=False, csp_objects=csp_objects)

print('-----MIBIF----')
pos, svm, lda, ss = train_mibif(X_full, y_full)
eval_mibif(pos, svm, lda, ss, X_eval, y_eval)

print('------RF------')
clf, ss = train_rf(X_full, y_full)
eval_rf(clf, ss, X_eval, y_eval)

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
csp_objects, X_full, y_full = preprocess_david(train_df, asynch_label, learning_method='standard', train=True)
X_eval1, y_eval1 = preprocess_david(eval_df1, eval_df1.iloc[:, -1].values, learning_method='standard', train=False, csp_objects=csp_objects)
X_eval2, y_eval2 = preprocess_david(eval_df2, eval_df2.iloc[:, -1].values, learning_method='standard', train=False, csp_objects=csp_objects)

print('-----MIBIF----')
pos, svm, lda, ss = train_mibif(X_full, y_full)
eval_mibif(pos, svm, lda, ss, X_eval1, y_eval1)
eval_mibif(pos, svm, lda, ss, X_eval2, y_eval2)

print('------RF------')
clf, ss = train_rf(X_full, y_full)
eval_rf(clf, ss, X_eval1, y_eval1)
eval_rf(clf, ss, X_eval2, y_eval2)