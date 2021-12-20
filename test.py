import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pl

def desnormalizar_y(y, max_gasto, max_coste):
    gasto = y[:,0]*max_gasto
    coste = y[:,1]*max_coste
    return gasto, coste

def normalizar_x(year, month):
    x = np.column_stack(((year - 2015.0) / 10, (np.cos(month * 2 * np.pi / 12) + 1) / 2, (np.sin(month * 2 * np.pi / 12) + 1) / 2))
    return x

if __name__ == '__main__':
    network = tf.keras.models.load_model('my_nn.h5')

    max_gasto = 263643.0
    max_coste = 10801.65

    year = []
    month = []
    for i in range(12):
        year = np.append(year, 2021)
        month = np.append(month, i+1)

    x = normalizar_x(year, month)

    y = network.predict(x)
    gasto, coste = desnormalizar_y(y, max_gasto, max_coste)

    for i in range(len(x)):
        print("fecha = ", int(month[i]), "/", int(year[i]), "gasto = ", gasto[i], ", coste = ", coste[i])


