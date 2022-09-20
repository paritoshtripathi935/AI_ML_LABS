# implement multilinear regression without using any libraries on Lab_3.1/student-por.csv

import csv
import numpy as np
import matplotlib.pyplot as plt

# read data from csv file
def read_data(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

# convert data to numpy array
def convert_to_numpy(data):
    data = np.array(data)
    return data

# convert data to float
def convert_to_float(data):
    data = data.astype(np.float)
    return data


# split data into train and test
def split_data(data):
    train = data[:int(len(data)*0.8)]
    test = data[int(len(data)*0.8):]
    return train, test


# calculate mean
def mean(data):
    return np.mean(data, axis=0)


# calculate standard deviation
def std(data):
    return np.std(data, axis=0)


# implement mlr
def mlr(train, test):
    # calculate mean and std
    train_mean = mean(train)
    train_std = std(train)

    # normalize data
    train = (train - train_mean) / train_std  # normalize train data
    test = (test - train_mean) / train_std

    # add bias
    train = np.insert(train, 0, 1, axis=1) # add bias to train data by adding a column of 1s
    test = np.insert(test, 0, 1, axis=1)  # add bias to test data by adding a column of 1s

    # split train and test data
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]

    # calculate weights using normal equation method (w = (X^T * X)^-1 * X^T * y) 
    weights = np.dot(np.dot(np.linalg.inv(np.dot(train_x.T, train_x)), train_x.T), train_y)


    # calculate predictions using test data and weights calculated above (y = X * w) using dot product
    predictions = np.dot(test_x, weights)
    # print predictions and test_y
    print(predictions)
 
    # calculate error using mean squared error (MSE) (MSE = 1/n * sum((y - y_hat)^2))
    error = np.sum(np.square(predictions - test_y)) / len(test_y)

    print('y = {}x + {}'.format(weights[1], weights[0]))

    return error


# main function
def main():
    # read data
    data = read_data('student-por.csv')

    # convert data to numpy array
    data = convert_to_numpy(data)

    # convert data to float
    data = convert_to_float(data)

    # split data into train and test
    train, test = split_data(data)

    # implement mlr
    error = mlr(train, test)

    # print error
    print(error)

    # plot error
    plt.plot(error)
    plt.show()

    # predict using mlr
    predict = mlr(train, test)




if __name__ == '__main__':
    main()





