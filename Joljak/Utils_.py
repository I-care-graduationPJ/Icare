import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Data_info

# 해당 조건을 만족하는 모든 파일들의 경로를, list로 묶어서 반환
def getFilePaths(abs_path: str, rel_path: str, sensor_type: str, date_str=None, time_start=None, time_end=None)->list:
    '''
    Returns a list of filepaths specified by date and time
    The first number indicates the activity. (ex-0:write, 1:lifting-hands, etc..)

    :param sensor_type: string - 'Acc', 'Gyro'
    :param date_str: date - '11.25'
    :param time_start: '10-11'
    :param time_end:
    :return: a list of file paths
    '''

    if sensor_type not in Data_info.SENSOR_TYPE:
        raise Exception("sensor_type parameter error: please provide correct str value")

    if sensor_type == 'Acc/AccZ':
        sensor_type = 'Acc'

    # rel_path = 'files/'
    # path = r'C:\Users\USER\Desktop\Machine_Learning\File\files'

    from os import listdir
    from os.path import isfile, join
    # extract file names
    onlyFiles = [f for f in listdir(abs_path) if isfile(join(abs_path, f))]

    # print("Inside [getFilesList] function:")
    # print(onlyFiles)
    sensor_type = sensor_type + '_'

    if date_str!=None:
        sensor_type = sensor_type + "[" + date_str + "]"

        if time_start!=None:
            sensor_type = sensor_type + " " + time_start

    vals = [str for str in onlyFiles if str[1:].startswith(sensor_type)]
    final_val = [rel_path + val for val in vals]

    # filePaths = [rel_path + filename for filename in onlyFiles]
    return final_val


# 특정 센서 타입만 얻음.
# "sensor_type" 이름이 붙은 모든 파일들 다 끌어모음.
def getAllFileNames(folder_dir: str, sensor_type: str, rel_path: str = None) -> list :
    '''
    이미 맨앞에 0,1,2 등 Activity가 레이블링 되있는 파일들.
    파일이름이 'Acc_'로 시작하는 파일이름 모두 Return (List형태)

    :param folder_dir: 파일들이 위치한 폴더경로
    :param rel_path:
    :return: A list with all 'Acc_' fileNames that are included in that directory
    '''

    if (type(sensor_type) != str) or (sensor_type not in ['Acc/AccZ', 'Gyro']):
        raise Exception('Utils_.getAllFileNames 함수 에러 -> sensor_type 인자')

    from os import listdir
    from os.path import isfile, join

    if sensor_type == 'Acc/AccZ':
        sensor_type = sensor_type[:-5]

    # Extract Acc file names (listidr - list all files in that directory)
    fileNames = [f for f in listdir(folder_dir) if isfile(join(folder_dir, f)) and f[1:].startswith(sensor_type + '_')]
    result = fileNames

    # Add rel path so that path is also added.
    if rel_path is not None:
        result = [rel_path + fn for fn in fileNames]

    return result



# 파일 이름들의 목록들을, 상위폴더와 결합하여 Path(list) 반환
def convert2Paths(rel_path:str , fileNames:list)->list:
    return [rel_path + fileName for fileName in fileNames]

# def getDataFrameInfo(df:pd.DataFrame)->dict:


# Mobile
# used in Chaquopy - real-time processing
def getDataFrameFromStr(acc_data:str, getDfInfo:bool = True)->tuple:
    '''

    :param acc_vals: String which Accel vals are recorded with timestamps
    :param getDfInfo: if True, return SamplingFrequency and period also.
    :return: dataframe made out of this string
    '''
    # split the string with delimiter='\n'
    acc_list = acc_data.split('\n')


    timestamp=[]
    x_val = []
    y_val = []
    z_val = []

    for one_row in acc_list:
        one_row_list = one_row.split(',')

        timestamp.append(int(one_row_list[0]))
        x_val.append(float(one_row_list[1]))
        y_val.append(float(one_row_list[2]))
        z_val.append(float(one_row_list[3]))
    data_dict = {'timestamp': timestamp, 'x_data': x_val, 'y_data': y_val, 'z_data': z_val}
    df: pd.DataFrame = pd.DataFrame(data_dict)

    # time range (duration),

    if getDfInfo == True:
        # Sampling rate(fs) : used for filters later
        period = (df['timestamp'][len(df) - 1] - df['timestamp'][0]) / 10 ** 9  # period (in seconds)
        # Sample rate
        fs = len(df) / period  # sample rate, Hz ('fs' number of data per second)

        return df, fs, period
    else:
        return df





# 제공된 파일로부터, (Dataframe 파일, 행동종류(숫자), Sampling Frequency, duration) 리턴.
def getDF_xyz(file_path_rel: str, rel_path: str, getDfInfo:bool=True)-> tuple:
    '''
    Convert text file (raw sensor data) to a dataframe with ['timestamp', 'x_data', 'y_data', 'z_data'] columns
    :param file_path_rel: string of the filepath
    :return:pd.DataFrame, datframe_info(fs, period)
    '''

    # read data
    data_array: np.array = np.loadtxt(file_path_rel, delimiter=',')
    data_array = list(data_array)
    data_array = [list(i) for i in data_array]

    timestamp = []
    x = []
    y = []
    z = []

    for i in data_array:
        timestamp.append(int(i[0]))
        x.append(float(i[1]))
        y.append(float(i[2]))
        z.append(float(i[3]))

    # timestamp.sort()

    data_dict = {'timestamp': timestamp, 'x_data': x, 'y_data': y, 'z_data': z}
    df: pd.DataFrame = pd.DataFrame(data_dict)


    starting_index:int = len(rel_path)
    label: int = int(file_path_rel[starting_index])

    # time range (duration),

    if getDfInfo == True:
        # Sampling rate(fs) : used for filters later
        period = (df['timestamp'][len(df) - 1] - df['timestamp'][0])/10**9  # period (in seconds)
        # Sample rate
        fs = len(df) / period  # sample rate, Hz ('fs' number of data per second)

        return df, label, fs, period
    else:
        return df



# text file -> DF : convert
def getDF_x(file_path: str) -> pd.DataFrame:

    text_file_string = np.loadtxt(file_path, delimiter=',')
    text_file_string = list(text_file_string)
    text_file_string = [list(i) for i in text_file_string]

    timestamp = []
    rate = []

    for i in text_file_string:
        timestamp.append(int(i[0]))
        rate.append(int(i[1]))

    # timestamp.sort()
    data_dict = {'timestamp': timestamp, 'rate': rate}
    df = pd.DataFrame(data_dict)
    return df

# 원본 데이터 출력
def plot_raw_data(time: pd.Series, raw_data: pd.DataFrame, axis: int = None, color: str = 'k') -> None:
    '''
    한 축의 데이터를 Plot 해줌.

    :param X_time: 1-axis time-data for raw-data.
    :param raw_data: 1-axis raw data
    :param axis: 0:x, 1:y, 2:z
    :param color: line color
    :return: nothing
    '''
    label_kinds = ['X', 'Y', 'Z']
    if axis == 0:
        color = 'r'
    elif axis == 1:
        color = 'g'
    elif axis == 2:
        color = 'b'

    plt.figure(num = 'Raw ' + label_kinds[axis] + '-Axis' + ' sensor data')
    plt.plot(time, raw_data, color, linewidth=1, label=label_kinds[axis] + '-Axis')
    plt.xlabel('Time [sec]')
    plt.ylabel('Sensor Value')
    # plt.grid()
    plt.legend()
    plt.show()
    
# 저주파필터 적용한 데이터와, 원본raw데이터를 출력해서 비교.
# Compare-plot with the raw data (option)
def plot_lpf(time:pd.Series, filtered_data: pd.DataFrame, axis: int, cutoff: float = 0.5, raw_data: pd.DataFrame = None) -> None:
    '''
    Plot filtered data

    :param time: time-data for filtered data.
    :param filtered_data: Filtered data
    :param axis: X-0, Y-1, Z=2
    :param cutoff:
    :param raw_data: default=None, If provided, compare
    :return:
    '''

    # axis 조건 체크
    if not [0, 1, 2].__contains__(axis):
        print('plot_lpf error! Parameter axis is not in [0,1,2]')
        print('plot_lpf, line103, preprocessing')

    label_axis = ['X', 'Y', 'Z']
    color = ['r', 'g', 'b']

    plt.figure(num='Filtered ' + label_axis[axis] + "-Axis data - " + str(cutoff) + "Hz cut-off Low-pass filter")
    plt.plot(time, filtered_data, color[axis], linewidth=2, label='filtered')

    # if a dataframe is given to raw_data:
    if type(raw_data) == pd.core.frame.Series:
        plt.plot(time, raw_data, 'k', linewidth=1, label='raw')


    plt.xlabel('Time [sec]')
    plt.ylabel('Sensor Value')
    # plt.grid()
    plt.legend()
    plt.show()

# After smoothing x,y,z -> GET vector-magnitude
def get_norm(*columns)->list:

    # columns : is a package of np.arrays (np.arrays of X, Y ,Z data)
    # Index: 0,1,2 = X, Y, Z

    if len(columns) != 3:
        print('get_norm() Error: must input three parameters')
        pass

    # zip X, Y, Z data together, make it an element
    vector_list = list(zip(columns[0], columns[1], columns[2]))  # zip x,y,z values and store them in a list
    # get vector magnitude, store them in a list
    norm = [np.linalg.norm(vec) for vec in vector_list]
    return norm


# generate a plot(xyz data)
def plot_xyz(df: pd.DataFrame, ylabel_string: str) -> None:
    '''
    :param df: datafrme
    :param df2: datafrme to plot with df
    :param ylabel_string: y label-name for plot
    :return: nothing
    '''
    plt.figure(num=ylabel_string + 'raw data', figsize=[15, 35])
    # plt.plot('timestamp', 'x_data', data=df, marker='o', markerfacecolor='blue', markerstze=12, color='skyblue', linewidth=2, label="x")
    plt.plot('timestamp', 'z_data', data=df, color='b', linewidth=1, label="z")
    plt.plot('timestamp', 'x_data', data=df, color='r', linewidth=1, label="x")
    plt.plot('timestamp', 'y_data', data=df, color='g', linewidth=1, label="y")

    plt.title(ylabel_string + ": x, y, z with respect to time")
    plt.legend(loc=2)
    plt.xlabel('Time[sec]')
    plt.ylabel(ylabel_string)
    plt.show()

# # *df -> receive multiple argments and pack them as a tuple
# def plot_one(xlabel_string, ylabel_string, *df):
#
#     plt.figure(num=ylabel_string + 'raw data', figsize=[15, 35])
#     # plt.plot('timestamp', 'x_data', data=df, marker='o', markerfacecolor='blue', markerstze=12, color='skyblue', linewidth=2, label="x")
#     # plt.plot('timestamp', 'z_data', data=df, color='b', linewidth=1, label="z")
#     plt.plot('timestamp', 'x_data', data=df, color='b', linewidth=1, label="Heart rate")
#
#     plt.plot(t, data, 'b-', label='data')
#     plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
#
#     plt.title(ylabel_string + ": x, y, z with timestamp")
#     plt.legend(loc=2)
#     plt.xlabel('time(s)')
#     plt.ylabel(ylabel_string)
#     plt.show()

def getDF_heart(HeartRate_data):
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

def plot_heart():
    pass

