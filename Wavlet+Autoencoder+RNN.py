import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt2

# read the data set
spy = pd.read_csv('Exxon.csv')


# The function is used to standardize the 3D data
def standard_scaler(X):
    samples, nx, ny = X.shape
# make X 2D and then use function StandardScaler
    X = X.reshape((samples, nx * ny))
    preprocessor = prep.StandardScaler().fit(X)
    X = preprocessor.transform(X)
# remake X 3D
    X = X.reshape((samples, nx, ny))
    return X


# The function is to preprocess the raw data
def preprocess_data(df, seq_len):
    features = len(df.columns)
    data = df.as_matrix()
# Get recurrent matrix for RNN, seq_len is the recurrent length
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
# The first 90% rows are training set
    result = np.array(result)
    train_row = round(0.9 * result.shape[0])
    train = result[: int(train_row), :]
# Make the data standard
    train = standard_scaler(train)
    result = standard_scaler(result)
# The row of X is from 0:-1 and the row of y is from 1:
# The day of X is a day before the corresponding y
    X_train = train[:-1, : -1]
    y_train = train[1:, -1][:, -1]
    X_test = result[int(train_row):-1, : -1]
    y_test = result[int(train_row)+1:, -1][:, -1]
# Remake the data proper dimensions
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], features))

    return [X_train, y_train, X_test, y_test]


# RNN and LSTM are used to modeling
def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        # This is the number of LSTM neurons in one hidden layer
        10,
        # This is the training dimension for LSTM
        input_shape=(layers[1], layers[0]),
        # Setting True make us add another stack LSTM
        return_sequences=True))
# Dropout is a regularization method, the number is the ratio to discard randomly
    model.add(Dropout(0.4))
# This is the input shape
    model.add(Dense(layers[1]))
# we stack another LSTM layer
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.35))

    model.add(Dense(layers[3]))
# choose the activation method to control whether LSTM block is active
    model.add(Activation("linear"))
# get the running time
    start = time.time()
# choose the method to optimize the model
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
# print the running time
    print("Compilation Time : ", time.time() - start)
    return model


# use the function to know the stock going up or down every day
def get_sign(y):
    signs = np.zeros(len(y)-1)
    for i in range(len(y)-1):
        signs[i] = np.sign(y[i+1]-y[i])
    return signs

# set LSTM recurrent block has 10 days
recurrent = 10

# make the raw day to training and testing data
X_train, y_train, X_test, y_test = preprocess_data(spy[: -1], recurrent)

# build the model
model = build_model([X_train.shape[2], recurrent, 100, 1])
model.fit(
    X_train,
    y_train,
    # choose how many sample will be updated
    batch_size=500,
    # choose how many epochs to train the model
    epochs=300,
    # choose the ratio to do validation
    validation_split=0.1,
    # choose how much results will be shown
    verbose=0)

# print the evaluation score for training set
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE' % trainScore[0])

# print the evaluation score for testing set
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE' % (testScore[0]))

diff = []
ratio = []

# get the predicted values
pred = model.predict(X_test)

# get the error value and error ratio
for u in range(len(y_test)):
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))

# get the increasing and decreasing directions
prep_updown = get_sign(pred)
true_updown = get_sign(y_test)

# get the ratio of true predictions
updown_ratio = sum(prep_updown == true_updown)

# print the value of the ratio of correctness
print("The ratio of correctness is %.2f " % updown_ratio)

# plot the predicted values versus true values
plt2.plot(pred, color='red', label='Prediction')
plt2.plot(y_test, color='Green', label='True Curve')
plt2.legend(loc='upper left')
plt2.show()
