import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

np.random.seed(1234)


def read_data():
    yelp_data = pd.read_csv("./Data/final_df.csv",
                            index_col=0, parse_dates=[1])
    yelp_data.sort_values(by='Period_StartDate', inplace=True)

    X_data = yelp_data.drop(['Open_Flag'], axis=1)  # Features
    y_data = yelp_data.Open_Flag  # Target variables

    sub_ind = X_data[X_data.Period_StartDate > '12/31/2015']
    X_test = X_data.ix[sub_ind.index, ]  # Subset for last 6 months
    y_test = y_data.ix[sub_ind.index, ]  # Subset for last 6 months

    X_data.drop(['Period_StartDate'], axis=1, inplace=True)
    X_data.drop(sub_ind.index, inplace=True)

    return (X_data.values, y_data.values, X_test.values, y_test.values)


def build_model():
    model = Sequential()
    layers = [100, 100, 1]  # Sizes of the layers

    model.add(LSTM(input_shape=(30, 23), output_dim=layers[
              0], return_sequences=False))
    model.add(Dropout(0.2))
    # model.add(LSTM(layers[2], return_sequences=False))
    # model.add(Dropout(0.2))
    # model.add(Dense(output_dim=layers[3]))
    model.add(Activation("sigmoid"))
    start = time.time()
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    print("Compilation Time : ", time.time() - start)
    return (model)


def run_network():
    global_start_time = time.time()
    epochs = 1
    # seq_length = 90
    # dataset_path = './Data/final_df_LSTM.csv'

    X_train, y_train, X_test, y_test = read_data()

    model = build_model()

    model.fit(X_train, y_train, batch_size=512,
              nb_epoch=epochs, validation_split=0.05)
    # predicted = model.predict(X_test)
    # predicted = np.reshape(predicted, (predicted.size,))

    scores = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print('Training duration (s) : ', time.time() - global_start_time)
    return (model, y_test, scores)


if __name__ == '__main__':
    model, y_test, predicted = run_network()
    # X_train, y_train, X_test, y_test = read_data()
