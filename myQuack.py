"""

Some partially defined functions for the Machine Learning assignment.

You should complete the provided functions and add more functions and classes
as necessary.

Write a main function that calls different functions to perform the required
tasks.

"""
import numpy as np
from sklearn import svm, neighbors, tree
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
    # Add if the tumour is benign or malignant into the 'y' list.
    y_list = [1 if row[1] == b'M' else 0 for row in data]
    # Copy the row, except ID and benign status to the new 'x' list. ID is
    # irrelevant to if it's cancerous or not.
    X_list = [list(row)[2:] for row in data]

    return np.array(X_list), np.array(y_list)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
    ##         "INSERT YOUR CODE HERE"
    raise NotImplementedError()


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
    dt_classifier = tree.DecisionTreeClassifier()
    params = [
        {
            'splitter': ['best', 'random'],
            'max_depth': np.linspace(1, 100, 100)
        }
    ]
    clf = GridSearchCV(dt_classifier, params)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Constants for the Nearest Neighbor classifier
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
    neighbor_classifier = neighbors.KNeighborsClassifier()
    # Use number of neighbours, and size of leaves as main parameters
    params = [
        {
            'n_neighbors': np.arange(MAX_N_NEIGHBORS) + 1,
            'leaf_size': np.arange(MAX_LEAF_SIZE) + 1
        }
    ]
    clf = GridSearchCV(neighbor_classifier, params)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


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
    svm_classifier = svm.SVC()
    params = [
        {
            'C': np.logspace(-3, 3, 7),
            'kernel': ['linear']
        },
        {
            'C': np.logspace(-3, 3, 7),
            'gamma': np.logspace(-4, 4, 9),
            'kernel': ['rbf']
        }
    ]
    clf = GridSearchCV(svm_classifier, params)
    clf.fit(X_training, y_training)
    return clf


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

TEST_SPLIT = 0.3

if __name__ == "__main__":
    start_time = time.clock()
    # Create initial training and testing data set.
    X, y = prepare_dataset('medical_records.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SPLIT)

    # Create a list of classifiers with names.
    classifiers = [[build_NB_classifier, "Naive Bayes"],
                   [build_DT_classifier, "Decision Tree"],
                   [build_NN_classifier, "Nearest Neighbour"],
                   [build_SVM_classifier, "Support Vector Machine"]]

    # Test each classifier and output values.
    for classifier_function, name in classifiers:
        classifier = classifier_function(X_train, y_train)
        print(name + " best params: ", classifier.best_params_)
        print(name + " score: %0.2f\n" % classifier.score(X_test, y_test))

    # Output time taken for overall testing
    print("\nTook %0.2f seconds to run" % (time.clock() - start_time))
