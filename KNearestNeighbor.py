import pandas as pd
import numpy as np
from collections import Counter
# train_test_split is a function to easily split data into training and testing sets.
from sklearn.model_selection import train_test_split
# confusion_matrix and accuracy_score are used to evaluate the performance of our model.
from sklearn.metrics import confusion_matrix, accuracy_score
# MinMaxScaler is used to normalize our feature data to a common scale [0, 1].
from sklearn.preprocessing import MinMaxScaler

# --- 2. Load and Prepare the Data ---
# Create a list of column names for our dataframe, as the CSV file doesn't have a header row.
# We create 30 feature names ('f1', 'f2', ...) and one 'class' label name.
columns_names = [f'f{i}' for i in range(1, 31)] + ['class']
# Load the dataset from the CSV file into a pandas DataFrame, assigning our custom column names.
df = pd.read_csv('bcwdisc.data.mb.csv', header=None, names=columns_names)

# Separate the data into features (X) and the target label (y).
# X contains all columns EXCEPT 'class'. The axis=1 specifies we are dropping a column.
X = df.drop('class', axis=1)
# y contains ONLY the 'class' column.
y = df['class']

# --- 3. Split Data into Training and Testing Sets ---
# Split the dataset into 70% for training and 30% for testing.
# random_state=42 ensures that the split is the same every time we run the code (for reproducibility).
# stratify=y ensures that the proportion of class labels (-1 and 1) is the same in both the train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 4. Normalize the Feature Data ---
# Initialize the MinMaxScaler. This will scale all feature values to the range [0, 1].
# This is crucial for distance-based algorithms like kNN to prevent features with large ranges
# from dominating the distance calculation.
scaler = MinMaxScaler()

# Fit the scaler ONLY on the training data. This calculates the min and max for each feature
# from the training set. We do this to avoid "data leakage" from the test set.
scaler.fit(X_train)

# Use the fitted scaler to transform both the training and testing data.
# The same scaling parameters (min/max from training) are applied to both sets.
X_train_normalized = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Re-combine the normalized features with their corresponding labels.
# This makes it easier to pass rows containing both features and the label to our functions.
# .reset_index(drop=True) is needed to ensure the indices align correctly after the split.
train_data = pd.concat([X_train_normalized, y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test_normalized, y_test.reset_index(drop=True)], axis=1)

# --- 5. Define the Core kNN Functions ---

# This function calculates the Euclidean distance between two data points (rows).
def euclidean_distance(point1, point2):
    # Select all values in the row except the last one (which is the class label).
    features1 = point1[:-1]
    features2 = point2[:-1]
    # Calculate the squared difference for each feature, sum them, and take the square root.
    return np.sqrt(np.sum((features1 - features2) ** 2))

# This function predicts the class for a single, unknown instance from the test set.
def predict_classification(train_data, test_instance, k):
    # This list will store tuples of (training_point, distance_to_test_point).
    distances = []
    
    # Loop through each row in the training data.
    for index, train_row in train_data.iterrows():
        # Calculate the distance between the current training row and the test instance.
        distance = euclidean_distance(test_instance, train_row)
        # Add the training row and its calculated distance to our list.
        distances.append((train_row, distance))

    # Sort the list of distances in ascending order. The `key` tells sort to use the second element
    # of each tuple (the distance) for sorting.
    distances.sort(key=lambda tup: tup[1])

    # Get the top 'k' entries from the sorted list. These are the k-nearest neighbors.
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    
    # Extract just the class labels from our list of neighbors.
    neighbor_labels = [row['class'] for row in neighbors]

    # "Vote" for the most common class label among the neighbors.
    # Counter creates a count of each label, and .most_common(1) gets the most frequent one.
    prediction = Counter(neighbor_labels).most_common(1)[0][0]
    return prediction

# This function runs the prediction process for the entire test set.
def k_nearest_neighbors(train_data, test_data, k):
    predictions = []
    # Loop through each row in the test data.
    for index, test_row in test_data.iterrows():
        # Call our prediction function for the current test row.
        prediction = predict_classification(train_data, test_row, k)
        # Add the resulting prediction to our list of predictions.
        predictions.append(prediction)
    return predictions

# --- 6. Run the kNN Algorithm and Evaluate Results ---

# Define the different values of 'k' we want to test.
k_values = [1, 3, 5, 7, 9]

print("📊 kNN Classifier Results 📊\n")

# Loop through each specified k value.
for k in k_values:
    # Get the list of predictions for the entire test set using our custom kNN function.
    y_pred = k_nearest_neighbors(train_data, test_data, k)
    
    # Calculate the accuracy by comparing the true labels (y_test) with our predictions (y_pred).
    accuracy = accuracy_score(y_test, y_pred)
    # Generate the confusion matrix to see the breakdown of true/false positives/negatives.
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])

    # Print the formatted results for the current value of k.
    print(f"--- Results for k = {k} ---")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("Confusion Matrix:")
    print("             Predicted: 1 | Predicted: -1")
    # TP (True Positive), FN (False Negative)
    print(f"Actual: 1    [TP: {cm[0][0]:<5} | FN: {cm[0][1]:<5}]")
    # FP (False Positive), TN (True Negative)
    print(f"Actual: -1   [FP: {cm[1][0]:<5} | TN: {cm[1][1]:<5}]")
    print("-" * 30 + "\n")