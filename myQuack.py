"""

CAB320 - Artifical Intelligence
Assignment 2: Classification Task

Completed myQuack.py script for the Machine Learning assignment.

Authors:
    Madeline Miller (9342401),
    Lindsay Watt (8847436) &
    Astrid Jonelynas (8884731)

"""


import numpy as np
from sklearn import svm, neighbors, tree, naive_bayes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
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
    # Load data from file
    data = np.genfromtxt(dataset_path, delimiter=',', dtype=None)
    # Add if the tumour is benign or malignant into the 'y' list.
    y_list = [1 if row[1] == b'M' else 0 for row in data]
    # Copy the row, excluding ID and benign status to the new 'X' list.
    X_list = [list(row)[2:] for row in data]
    # Return the two numpy arrays X and y
    return np.array(X_list), np.array(y_list)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Parameter testing range for NB classifier
NB_PRIORS_START = 0.01
NB_PRIORS_STOP = 0.99
NB_PRIORS_NUM = 90


def build_NB_classifier(X_training, y_training):
    """
    Build a Naive Bayes classifier based on the training set X_training,
    y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    # Create GaussianNB classifier using sklearn library
    nb_classifier = naive_bayes.GaussianNB()
    # Use 'priors' as the main classifier parameter
    params = [
        {
            'priors': list(
                np.transpose(
                    [np.linspace(NB_PRIORS_START, NB_PRIORS_STOP, NB_PRIORS_NUM),
                     np.linspace(NB_PRIORS_STOP, NB_PRIORS_START, NB_PRIORS_NUM)])
            )
        },
        {
            'priors': [None]
        }
    ]
    # Estimate the best value of parameter using crossvalidated gridsearch
    clf = GridSearchCV(nb_classifier, params)
    # Train the model using the training data
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Parameter testing range for DT classifier
DT_DEPTH_START = 1
DT_DEPTH_STOP = 100
DT_DEPTH_NUM = 100


def build_DT_classifier(X_training, y_training):
    """
    Build a Decision Tree classifier based on the training set X_training,
    y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    # Create DT classifier using sklearn library
    dt_classifier = tree.DecisionTreeClassifier()
    # Use 'splitter' and 'max_depth' as the main classifier parameters
    params = [
        {
            'splitter': ['best', 'random'],
            'max_depth': np.linspace(DT_DEPTH_START, DT_DEPTH_STOP,
                                     DT_DEPTH_NUM)
        }
    ]
    # Estimate the best value of the parameters using crossvalidated gridsearch
    clf = GridSearchCV(dt_classifier, params)
    # Train the model using the training data
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Parameter testing range for NN classifier
MAX_N_NEIGHBORS = 20
MAX_LEAF_SIZE = 50


def build_NN_classifier(X_training, y_training):
    """
    Build a Nearest Neighbours classifier based on the training set
    X_training, y_training.

    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        clf : the classifier built in this function
    """
    # Create NN classifier using sklearn library
    neighbor_classifier = neighbors.KNeighborsClassifier()
    # Use 'n_neighbors' and 'leaf_size' as the main classifier parameters
    params = [
        {
            'n_neighbors': np.arange(MAX_N_NEIGHBORS) + 1,
            'leaf_size': np.arange(MAX_LEAF_SIZE) + 1
        }
    ]
    # Estimate the best value of the parameters using crossvalidated gridsearch
    clf = GridSearchCV(neighbor_classifier, params)
    # Train the model using the training data
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Parameter testing range for SVM classifier
SVM_C_START = -3
SVM_C_STOP = 3
SVM_C_NUM = 7
SVM_GAMMA_START = -4
SVM_GAMMA_STOP = 4
SVM_GAMMA_NUM = 9


def build_SVM_classifier(X_training, y_training):
    """
    Build a Support Vector Machine classifier based on the training set
    X_training, y_training.
    @param
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
    @return
        clf : the classifier built in this function
    """
    # Create SVM classifier using sklearn library
    svm_classifier = svm.SVC()
    # Use 'C', 'kernel' and 'gamma' as the main classifier parameters
    params = [
        {
            'C': np.logspace(SVM_C_START, SVM_C_STOP, SVM_C_NUM),
            'kernel': ['linear']
        },
        {
            'C': np.logspace(SVM_C_START, SVM_C_STOP, SVM_C_NUM),
            'gamma': np.logspace(SVM_GAMMA_START, SVM_GAMMA_STOP,
                                 SVM_GAMMA_NUM),
            'kernel': ['rbf']
        }
    ]
    # Estimate the best value of the parameters using crossvalidated gridsearch
    clf = GridSearchCV(svm_classifier, params)
    # Train the model using the training data
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Test data split value
TEST_SPLIT = 0.3


if __name__ == "__main__":
    # Print team member names
    print(my_team())
    # Start clock to allow for overall testing time to be computed
    start_time = time.clock()
    # Complete data pre-processing
    X, y = prepare_dataset('medical_records.data')
    # Create initial training and testing data set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SPLIT)

    # Create a list of classifiers with names
    classifiers = [[build_NB_classifier, "Naive Bayes"],
                   [build_DT_classifier, "Decision Tree"],
                   [build_NN_classifier, "Nearest Neighbour"],
                   [build_SVM_classifier, "Support Vector Machine"]]

    # Test each classifier and output values
    for classifier_function, name in classifiers:
        classifier = classifier_function(X_train, y_train)
        # Output best parameters for each classifier
        print(name, "Best Parameters:", classifier.best_params_)
        # Generate classification report for training data
        predictions_training = classifier.predict(X_train)
        print(name, "Training Data Classification Report:")
        print(classification_report(y_train, predictions_training))
        # Generate classification report for test data
        predictions = classifier.predict(X_test)
        print(name, "Test Data Classification Report:")
        print(classification_report(y_test, predictions))

    # Output time taken for overall testing
    print("\nTook %0.2f seconds to run" % (time.clock() - start_time))
