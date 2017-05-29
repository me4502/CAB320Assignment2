"""

Some partially defined functions for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes as necessary.

Write a main function that calls different functions to perform the required tasks.

"""
import numpy as np
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split, GridSearchCV
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    """
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    """
    return [(8884731, 'Astrid', 'Jonelynas'),
            (8847436, 'Lindsay', 'Watt'),
            (9342401, 'Madeline', 'Miller')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    """
    Read a comma separated text file where
        - the first field is a ID number
        - the second field is a class label 'B' or 'M'
        - the remaining fields are real-valued

    Return two numpy arrays X and y where
        - X is two dimensional. X[i,:] is the ith example
        - y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
        X,y
    """
    # load data from file
    data = np.genfromtxt(dataset_path, delimiter=',', dtype=None)
    y_list = []
    x_list = []

    for row in data:
        y_list.append(1 if row[1] == b'M' else 0)
        x_list.append(np.array(list(row)[2:]))
    return np.array(x_list), np.array(y_list)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_NB_classifier(X_training, y_training):
    """
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_DT_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_NN_classifier(X_training, y_training):
    """
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    neighbor_classifier = neighbors.KNeighborsClassifier()
    params = [
        {
            'n_neighbors': np.arange(20) + 1,
            'leaf_size': np.arange(50) + 1
        }
    ]
    clf = GridSearchCV(neighbor_classifier, params)
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def build_SVM_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set X_training, y_training.
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    """
    svm_classifier = svm.SVC()
    params = [
        {'C': np.logspace(-3, 3, 7), 'kernel': ['linear']},
        {'C': np.logspace(-3, 3, 7), 'gamma': np.logspace(-4, 4, 9), 'kernel': ['rbf']},
        # {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [1, 2]}
    ]
    clf = GridSearchCV(svm_classifier, params)
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    start_time = time.clock()
    X, y = prepare_dataset('medical_records.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    svc = build_NN_classifier(X_train, y_train)
    print("svc best params:", svc.best_params_)
    print("svc score: %0.2f" % svc.score(X_test, y_test))

    print("\nTook %d seconds to run" % (time.clock() - start_time))
