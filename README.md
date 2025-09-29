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

The first part of the analysis focused on applying K-Means to the kmtest dataset for K values of 2, 3, 4, and 5. A key objective was to observe the effect of Z-score normalization on the clustering outcome.
After applying Z-score normalization, the clustering algorithm was able to find meaningful groups in the data. This is because normalization made sure that all features were on the same scale, so features with larger values didn't unfairly influence the results. The figures show how this process helped the algorithm accurately identify clusters based on how close the data points actually are to one another.

<img width="427" height="319" alt="image" src="https://github.com/user-attachments/assets/49600b8b-12a6-40d3-99ff-6dfaf373876a" />
<img width="424" height="319" alt="image" src="https://github.com/user-attachments/assets/28ed278b-b413-48e7-a5d6-18c72f13b551" />
<img width="426" height="319" alt="image" src="https://github.com/user-attachments/assets/1fd8407f-493a-474b-83ef-3f61a1bf6d64" />
<img width="426" height="319" alt="image" src="https://github.com/user-attachments/assets/5697efff-58e8-4f3c-80b8-e3fde82651ba" />



The second analysis used the iris dataset to evaluate the algorithm's accuracy and stability. The algorithm was run five times with K=3. 
The five runs produced remarkably consistent and optimal outcomes, as shown by the Sum of Squared Errors (SSE) for each run:
    Run 1: SSE = 78.8514
    Run 2: SSE = 78.8557
    Run 3: SSE = 78.8557
    Run 4: SSE = 78.8514
    Run 5: SSE = 78.8514
The algorithm accurately separated the 'setosa' species and did a good job of dividing the overlapping 'versicolor' and 'virginica' species. A numerical analysis shows the algorithm is effective. The cluster centers it found were very close to the true centers, with an average distance of only 0.1727. This low value and the consistent, low scores across all tests prove the algorithm is stable and reliable on this dataset.

<img width="364" height="272" alt="image" src="https://github.com/user-attachments/assets/a52bbcb2-ea90-4a8a-934a-28f455ab22ae" />
<img width="364" height="274" alt="image" src="https://github.com/user-attachments/assets/7eb154b9-1889-4558-8a6a-92cbc14910bd" />
<img width="400" height="301" alt="image" src="https://github.com/user-attachments/assets/f75f7285-95d6-4803-9778-a05c25f2f3e2" />







