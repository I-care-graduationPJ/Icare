from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def read():
    HeartRate_data = np.loadtxt('HeartRate_data.txt', delimiter=',')
    HeartRate_data = list(HeartRate_data)
    HeartRate_data = [list(i) for i in HeartRate_data]

    timestamp =[]
    bpm = []

    for i in HeartRate_data:
        timestamp.append(int(i[0]))
        bpm.append(float(i[1]))

    data_dict = {'timestamp': timestamp, 'BPM': bpm}
    df = pd.DataFrame(data_dict)
    print(df)

    plt.figure(num='HeartRate data', figsize=[7, 10])
    # plt.plot('timestamp', 'x_data', data=df, marker='o', markerfacecolor='blue', markerstze=12, color='skyblue', linewidth=2, label="x")
    # plt.plot('timestamp', 'y_data', data=df, color='r', linewidth=2, label="y")
    # plt.plot('timestamp', 'y_data', data=df, color='r', linewidth=2, label="y")
    # plt.plot('timestamp', 'z_data', data=df, color='b', linewidth=1, label="z")
    plt.plot('timestamp', 'BPM', data=df, color='r', linewidth=1, label="bpm")
    # plt.plot('timestamp', 'y_data', data=df, color='g', linewidth=1, label="y")


    plt.title("HeartRate data:")
    plt.legend(loc=2)
    plt.xlabel('time')
    plt.ylabel('bpm')
    plt.show()

