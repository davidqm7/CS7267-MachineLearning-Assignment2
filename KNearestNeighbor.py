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

def euclidean_distance(point1, point2):
    features1 = point1[:-1]
    features2 = point2[:-1]
    return np.sqrt(np.sum((features1 - features2) ** 2))

def predict_classification(train_data, test_instance, k):
    distances = []
    
    for index, train_row in train_data.iterrows():
        distance = euclidean_distance(test_instance, train_row)
        distances.append((train_row, distance))

    distances.sort(key=lambda tup: tup[1])

    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    
    neighbor_labels = [row['class'] for row in neighbors]

    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    return prediction

def k_nearest_neighbors(train_data, test_data, k):
    predictions = []
    for index, test_row in test_data.iterrows():
        prediction = predict_classification(train_data, test_row, k)
        predictions.append(prediction)
    return predictions

k_value = [1, 3, 5, 7, 9]

print("KNN Classifier Results:")

for k in k_value:
    y_pred = k_nearest_neighbors(train_data, test_data, k)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])

    print(f"--- Results for k = {k} ---")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Confusion Matrix:")
    print("             Predicted: 1 | Predicted: -1")
    print(f"Actual: 1    [TP: {cm[0][0]:<5} | FN: {cm[0][1]:<5}]")
    print(f"Actual: -1   [FP: {cm[1][0]:<5} | TN: {cm[1][1]:<5}]")
    print("-" * 30 + "\n")


    
   
