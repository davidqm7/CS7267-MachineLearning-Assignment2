import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler


columns_names = [f'f{i}' for i in range(1, 31)] + ['class']
df = pd.read_csv('wdbc.data.mb.csv', header=None, names=columns_names)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = MinMaxScaler()

scaler.fit(X_train)

#training and testing data
X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

train_data = pd.concat([X_train_normalized, y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test_normalized, y_test.reset_index(drop=True)], axis=1)

