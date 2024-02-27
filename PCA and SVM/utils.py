import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    train_min,train_max = X_train.min(),X_train.max()
    test_min,test_max = X_train.min(),X_test.max()


    x_train_normalized = 2*((X_train-train_min)/(train_max - train_min )) - 1
    x_test_normalized = 2*((X_test-test_min)/(test_max - test_min )) - 1
    
    return x_train_normalized,x_test_normalized
    raise NotImplementedError

def plot_metrics(metrics) -> None:
    # plot and save the results
    components, accuracy, precision, recall, f1Score = [], [], [], [], []
    for i in range(len(metrics)):
        components.append(metrics[i][0])
        accuracy.append(metrics[i][1])
        precision.append(metrics[i][2])
        recall.append(metrics[i][3])
        f1Score.append(metrics[i][4])


    plt.plot(components, accuracy) 
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Graph ')
    plt.savefig("accuracy.jpg")
    plt.clf()

    plt.plot(components, precision) 
    plt.xlabel('Components')
    plt.ylabel('Precision')
    plt.title('Precision Graph ')
    plt.savefig("precision.jpg")
    plt.clf()

    plt.plot(components, recall) 
    plt.xlabel('Components')
    plt.ylabel('Recall')
    plt.title('Recall Graph ')
    plt.savefig("recall.jpg")
    plt.clf()

    plt.plot(components, f1Score) 
    plt.xlabel('Components')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Graph ')
    plt.savefig("f1score.jpg")
    plt.clf()


    # raise NotImplementedError