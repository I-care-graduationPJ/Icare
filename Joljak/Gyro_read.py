from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def read():

    Gyro_data = np.loadtxt('Gyro_data.txt', delimiter=',')
    Gyro_data = list(Gyro_data)
    Gyro_data = [list(i) for i in Gyro_data]

    timestamp = []
    x = []
    y = []
    z = []

    for i in Gyro_data:
        timestamp.append(int(i[0]))
        x.append(float(i[1]))
        y.append(float(i[2]))
        z.append(float(i[3]))

    data_dict = {'timestamp': timestamp, 'x_data': x, 'y_data': y, 'z_data': z}
    df = pd.DataFrame(data_dict)
    # print(df)

    plt.figure(num='Gyroscope data', figsize=[15,35])
    # plt.plot('timestamp', 'x_data', data=df, marker='o', markerfacecolor='blue', markerstze=12, color='skyblue', linewidth=2, label="x")
    plt.plot('timestamp', 'z_data', data=df, color='b', linewidth=1, label="z")
    plt.plot('timestamp', 'x_data', data=df, color='r', linewidth=1, label="x")
    plt.plot('timestamp', 'y_data', data=df, color='g', linewidth=1, label="y")


    plt.title("Gyroscope data")
    plt.legend(loc=2)
    plt.xlabel('time')
    plt.ylabel('Gyroscope data')
    plt.show()

