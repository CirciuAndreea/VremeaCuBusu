import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.python.keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
import tensorflow as tf


def get_data(df):
    X = df[["Month", "Humidity", "Pressure (millibars)"]].copy()
    Y = df[['Temperature (C)']].copy()

    return X, Y

def create_model(input_dim):
    beta = 0.5
    model = Sequential()

    model.add(Dense(8, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(beta))

    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(beta))

    model.add(Dense(1, activation='linear'))

    opt = Adam(learning_rate=0.01)
    model.compile(loss= 'mean_squared_error', metrics=['mean_squared_error'], optimizer=opt)

    return model


def run(df):
    print("Starting...")
    X, Y = get_data(df)
    print(X)


    #X = preprocessing.normalize(X)
    #Y = preprocessing.normalize(Y)
    print(Y)

    #Impartim datele
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    #print(Y_train)

    #Normalizarea datelor
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    scaler = StandardScaler().fit(Y_train)
    Y_train = scaler.transform(Y_train)
    Y_test = scaler.transform(Y_test)

    input_shape = 3
    model = create_model(input_shape)
    model.summary()

    try:
        model.load_weights("./checkpoint.h5")
    except OSError:
        print("No model checkpoint found")

    history = model.fit(X_train, Y_train, epochs=100, batch_size=30, shuffle=True)
    model.save('./checkpoint.h5')

    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print("Loss = " + str(loss))


    plt.plot(history.history['loss'])
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()