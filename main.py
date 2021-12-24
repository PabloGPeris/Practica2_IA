import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def is_number(string): # sacada de internet
    try:
        float(string)
        return True
    except ValueError:
        return False

def normalizar_x(year, month):
    x = np.column_stack(((year - 2015.0) / 10, (np.cos(month * 2 * np.pi / 12) + 1) / 2, (np.sin(month * 2 * np.pi / 12) + 1) / 2))
    return x

def normalizar_y(gasto, coste, max_gasto, max_coste):
    y = np.column_stack((gasto/max_gasto, coste/max_coste))
    return y

def desnormalizar_y(y, max_gasto, max_coste):
    gasto = y[:,0]*max_gasto
    coste = y[:,1]*max_coste
    return gasto, coste

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #****************************
    #***   lectura de datos   ***
    #****************************

    file = open("ETSII-Data-01.csv")
    csv_reader = csv.reader(file)

    month = np.array([], dtype = np.single)
    year = np.array([], dtype = np.single)
    gasto = np.array([], dtype = np.single)
    coste = np.array([], dtype = np.single)
    current_year = 2020

    for row in csv_reader:
        # columna 0 (meses/año)
        if is_number(row[0]): # es un año
            current_year = int(row[0]) # actualiza el año
            continue
        elif row[0] == "enero":
            month = np.append(month, 1.0)
        elif row[0] == "febrero":
            month = np.append(month, 2.0)
        elif row[0] == "marzo":
            month = np.append(month, 3.0)
        elif row[0] == "abril":
            month = np.append(month, 4.0)
        elif row[0] == "mayo":
            month = np.append(month, 5.0)
        elif row[0] == "junio":
            month = np.append(month, 6.0)
        elif row[0] == "julio":
            month = np.append(month, 7.0)
        elif row[0] == "agosto":
            if current_year == 2017: # mes eliminado manualmente
                continue
            month = np.append(month, 8.0)
        elif row[0] == "septiembre":
            month = np.append(month, 9.0)
        elif row[0] == "octubre":
            month = np.append(month, 10.0)
        elif row[0] == "noviembre":
            if current_year == 2016: # mes eliminado manualmente
                continue
            month = np.append(month, 11.0)
        elif row[0] == "diciembre":
            if current_year == 2015: # mes eliminado manualmente
                continue
            month = np.append(month, 12.0)
        else:
            continue

        # columna 1 (gasto de gas)
        if is_number(row[1]):
            if float(row[1]) < 0: # valores negativos se desechan. Hay que quitarlo
                month = np.delete(month, -1)
                continue
            gasto = np.append(gasto, float(row[1]))
        else:
            month = np.delete(month, -1) # mes sin datos no se cuenta
            continue

        # columna 3 (coste)
        if is_number(row[3]):
            if float(row[3]) < 0: # valores negativos se desechan. Hay que quitarlo
                month = np.delete(month, -1)
                gasto = np.delete(gasto, -1)
                continue
            coste = np.append(coste, float(row[3]))
        else:
            month = np.delete(month, -1) # mes sin datos no se cuenta
            gasto = np.delete(gasto, -1)
            continue
        year = np.append(year, current_year)

    file.close()

    for i in range(month.size):
        print(int(year[i]), " - ", month[i], ": ", gasto[i], "kWh, ", coste[i], "€")

    #*****************************************
    #***   Tratamiento  de datos para NN   ***
    #*****************************************

    max_gasto = np.max(gasto)*1.5
    max_coste = np.max(coste)*1.5
    print("max_gasto = ", max_gasto, "max_coste = ", max_coste)

    x = normalizar_x(year, month)
    y = normalizar_y(gasto, coste, max_gasto, max_coste)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    print(x_test)
    print(y_test)

    #*******************************
    #***   Entrenamiento de NN   ***
    #*******************************

    # Define the network by stacking layers
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu')
    ])

    # Training system
    network.compile(optimizer='sgd',
                    loss='mse',
                    metrics='mse')

    # Train the network with train data
    network.fit(x_train, y_train, epochs=250)

    # Evaluate the network on test data
    loss = network.evaluate(x_test, y_test)
    print('Loss =', loss)

    #**********************************
    #***   Guardado y test rápido   ***
    #**********************************

    network.save('my_nn.h5')

    y_pred = network.predict(x_test)
    gasto_pred, coste_pred = desnormalizar_y(y_pred, max_gasto, max_coste)
    gasto_test, coste_test = desnormalizar_y(y_test, max_gasto, max_coste)

    # ver resultados del test
    for i in range(len(x_test)):
        print("gasto test = ", gasto_test[i], ", gasto pred = ", gasto_pred[i], ", coste test = ", coste_test[i], ", coste pred = ", coste_pred[i], ".")
