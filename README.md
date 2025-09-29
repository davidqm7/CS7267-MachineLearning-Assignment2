# KNearestNeighbor Assignment Instructions

K-Nearest Neighbor (kNN) Classifier is a supervised pattern classifier that determines the class of
an input sample based on the distance to k nearest labeled neighbors. kNNs are considered as a
type of lazy learning method where an evaluation is performed as needed. In this assignment, you
are going to implement basic kNN algorithm and analyze the effect of normalization.
Implement basic kNN algorithm to classify the given data set. Do not use built-in function such as
knn. You must implement it by yourself. This is an individual assignment.

---

## Objective

Implement the basic K-Nearest Neighbor. 

**Note:** Do not use built-in functions. 

---

## Datasets

The sample data set to be used for this project is given below. The bcwdisc.data.mb.csv
is a version of a breast cancer database published by the University of Wisconsin Hospitals. The
original data set was obtained from the University of California Irvine (UCI) machine learning
repository (http://archive.ics.uci.edu/ml/ ). The data set is in .csv format, with one sample / pattern
vector per line of input. Each line will contain a series of attribute values, separated by commas.
The file does not contain headers. The first 30 columns are attributes, and the last attribute
indicates the class. The class labels are 1 (malignant) and -1 (benign). 

## Result

The k-Nearest Neighbor (kNN) classifier was successfully implemented and tested on the normalized Wisconsin Breast Cancer dataset for k-values of 1, 3, 5, 7, and 9. Accuracy measures the percentage of correctly classified samples, while the confusion matrix provides a detailed breakdown of correct (TP, TN) and incorrect (FN, FP) predictions for malignant (1) and benign (-1) classes. The figures below show the results of the k-values. 

<img width="465" height="245" alt="image" src="https://github.com/user-attachments/assets/836b9d51-8b8e-4e92-af85-706fcebb702f" />
<img width="468" height="247" alt="image" src="https://github.com/user-attachments/assets/f559cb78-46c1-408c-8633-a9340f2bb9fd" />
<img width="450" height="239" alt="image" src="https://github.com/user-attachments/assets/f8269d99-c12c-405a-b3da-c36cd20d9f27" />
<img width="445" height="239" alt="image" src="https://github.com/user-attachments/assets/56c8ff6a-45eb-42a4-a38e-7852e2745423" />
<img width="451" height="253" alt="image" src="https://github.com/user-attachments/assets/8a253d8b-90ca-4d32-b1eb-99b79c8f01ed" />
