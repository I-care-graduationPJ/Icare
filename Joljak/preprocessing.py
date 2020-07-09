from typing import Any, Union

# activity 설명:
# 0: writing
# 1: lifting

import numpy as np
import pandas as pd
import Utils_ as utils
from lowPassFilter import do_lpf
from scipy.signal import find_peaks
import bandPassFilter



# Extract primary features of a given raw-data dataframe
# If it is Acc, set Acc=True
def primary_feat(df: pd.DataFrame, Acc: bool = False, order=None)->dict:
    '''
    Accepts a raw data (dataframe) , and then refines it with low-pass-filter.
    And then, it creates a vector magnitude using that filtered data.

    :param df: DataFrame that contains raw data
    :param Acc: If True, it also returns the features of AccZ/AccMag. Otherwise, just Gyro features.
    :param order: An extra order that tells what the user wants in return.
    options: 'corr' - extract corr values of x,y,z
    :return: 1 dictionary - key:value --> feature names:feature_values

    '''

    # # Create a df from a textfile (entire data)
    # # df_accel = utils.getDF_xyz(file_path_acc)
    # print('Success')
    # print(df.head())

    # Timestamp -> convert to time in seconds that elapsed since start of the record.
    # df['timestamp'] = (df['timestamp'] - df['timestamp'][0]) / 10 ** 9

    # df.iloc[:,0] = (df.iloc[:,0] - df.iloc[0, 0])/(10**9)
    # print(df.head(5))

    # 타입 체크
    if type(df) != pd.DataFrame:
        print('param error: df must be a pd.Dataframe - 50줄')
        print('preprocessing.primary_feat function ends')
        return

    # Sampling rate(fs) : used for filters later
    # period = df['timestamp'][len(df) - 1] - df['timestamp'][0]  # period (in seconds)

    # Durtion
    period = (df.iloc[-1, 0] - df.iloc[0, 0])/10**9
    # print('period:', period)

    # Sample rate
    fs: float = len(df) / period  # sample rate, Hz ('fs' number of data per second)
    
    # print("len(df):", len(df))
    # print("fs: ", fs)
    cutoff = 0.5

    # separate data
    X_time: pd.Series = df['timestamp']
    X_data: pd.Series = df['x_data']
    Y_data: pd.Series = df['y_data']
    Z_data: pd.Series = df['z_data']


    # XYZ data
    XYZ_data = df.drop('timestamp', axis=1)


    # 데이터 정제
    # Apply lpf to original X, Y, Z data : 0.5 cutoff, 1st order - filtered data: np array
    if Acc is True:
        X_data_fil = do_lpf(X_data, fs, 3, 0.5)
        Y_data_fil = do_lpf(Y_data, fs, 3, 0.5)
        Z_data_fil = do_lpf(Z_data, fs, 3, 0.5)
    else: # it is Gyro
        X_data_fil = do_lpf(X_data, fs, cutoff=0.3)
        Y_data_fil = do_lpf(Y_data, fs, cutoff=0.3)
        Z_data_fil = do_lpf(Z_data, fs, cutoff=0.3)


    # 메소드로 변환하자.
    # Find correlation values, and return them as dict (the funciton ends)
    if order is 'corr':
        # params: two variables that we want to find the correlation coefficient
        corr_XY = np.corrcoef(X_data_fil, Y_data_fil)[0, 1]
        corr_YZ = np.corrcoef(Y_data_fil, Z_data_fil)[0, 1]
        corr_ZX = np.corrcoef(Z_data_fil, X_data_fil)[0, 1]

        return {'corr_XY': corr_XY, 'corr_YZ': corr_YZ, 'corr_ZX': corr_ZX}


    # After smoothing x,y,z, extract vector-magnitude of smoothed x,y,z / Returns norm of given vectors (list)
    def get_norm(*columns):

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

        # vector_list = list(zip(arr_Xdata, arr_Ydata, arr_Zdata))  # zip x,y,z values and store them in a list
        # # get vector magnitude, store them in a list
        # norm_accel = [np.linalg.norm(vec) for vec in vector_list]
        #
        # return norm_accel # list


    # Get Norm
    VecMag = utils.get_norm(X_data_fil, Y_data_fil, Z_data_fil)

    # Get AccZ/AccMag
    AccZ_scaled = Z_data_fil / VecMag


    ### 2nd order feature(1)
    # apply lpf on norm
    VecMag_2ndFeature_1 = do_lpf(VecMag, fs)  ##### fianl data
    # plot_norm(SecondFeature_1, 'filtered')

    AccZ_2ndFeature_1 = do_lpf(AccZ_scaled, fs)

    # norm_fil: 1d-array, threshold

    # default threshold=8.5
    # peak_detect(norm_fil)

    if Acc == True:
        # rename data to Acc
        AccMag = VecMag
        AccMag_2ndFeature_1 = VecMag_2ndFeature_1

        # 수정 - key값 수정
        # dict form
        Features = {'First_features': {'AccMag': AccMag, 'AccZ/AccMag': AccZ_scaled},
                    'Second_features': {'AccMag_2_1': AccMag_2ndFeature_1, 'AccZ_2_1': AccZ_2ndFeature_1}}

        ## tuples of lists form
        # feature_data = AccMag, VecMag_2ndFeature_1, AccZ_scaled, AccZ_2ndFeature_1

        # mean, sd, energy, correlation, freq-domain entropy


    else:
        # rename data to 'Gyro'
        GyroMag = VecMag
        GyroMag_2ndFeature_1 = VecMag_2ndFeature_1

        # dict form
        Features = {'First_features': {'GyroMag': GyroMag},
                    'Second_features': {'GyroMag_2_1': GyroMag_2ndFeature_1}}

        # # tuple of lists form
        # feature_data = GyroMag, GyroMag_2ndFeature_1


    return Features


# Return mean, std, sum of an array
def extract_feat_mss(arr:np.array)->tuple:
    if not type(arr) is np.array:
        arr = np.array(arr)
    return arr.mean(), arr.std(), arr.sum()


# 이미 레이블링된 파일에 대한 경로 제공한다.
# 상대경로: 이 .py파일 입장에서의 경로
# Caution: This uses preprocessing.primary_feat (primary features like norms, filtered norms, etc..)
# Gyro도 자동으로 path 제공

def extract_feature_row(file_path_Acc: str, rel_path_len: str, win_size: object = '2s', label: object = None) -> dict:
    '''
    Extract features as a dict (representing 1 row in dataframe), which contains
    all sensors (Acc, Gyro).

    You just need to provide relative file_path_Acc, which will automatically generate filepaths for
    Gyro.

    *Caution: This severs the first 1-second and the last 1-second of the data, because the sensor values
    are likely to be irrelevant to the activity.

    :param file_path_Acc: relative path for one file

    :param 'win_size': String indicating the size of the window in seconds - ex)'1s', '0.5s'
    :param file_path_Gyro: modified from file_path_Acc
    :param label: Activity string - 0:write, 1:lift
    :return: a dict that represents a row of a Dataframe. It contains the label column also.
    '''
    # file_path_Acc = 'files/1Acc_[10.31] 10-20-16...10-20-32.txt'
    # file_path_Gyro = 'files/1Gyro_[10.31] 10-20-16...10-20-32.txt'

    # length of the prefix : len of relative path = len('files/') = 6
    # where the label is.
    starting_index:int = rel_path_len

    # Exclude the prefix, and get the label result in integer (from string)
    label:int = int(file_path_Acc[starting_index])

    # only replace this part of the string.
    file_path_Gyro = file_path_Acc.replace('Acc_', 'Gyro_')

    # Dataframe 생성 (for this filepath), df_Acc_info : 해당 파일의 samplingRate, period 정보
    df_Acc, df_Acc_info = utils.getDF_xyz(file_path_Acc)

    # 1초에 몇개의 instances?
    fs_acc = int(df_Acc_info['fs'])

    # Exclude 1sec from the start and the end
    df_Acc = df_Acc[fs_acc:-fs_acc]


    # For Gyro: get DF from textfile
    df_Gyro, df_Gyro_info = utils.getDF_xyz(file_path_Gyro)

    # 1초에 몇개의 instances?
    fs_gyro = int(df_Gyro_info['fs'])
    df_Gyro = df_Gyro[fs_gyro:-fs_gyro]

    # Primary Feature Extraction
    primary_feat_info:dict = primary_feat(df_Acc, Acc=True)
    acc_features = primary_feat_info

    AccMag = np.array(acc_features['First_features']['AccMag'])
    AccMag_2_1 = np.array(acc_features['Second_features']['AccMag_2_1'])

    AccZ_scaled = np.array(acc_features['First_features']['AccZ/AccMag'])
    AccZ_scaled_2_1 = np.array(acc_features['Second_features']['AccZ_2_1'])

    # Correlation, using filtered Acc X,Y,Z data : 3
    corr_dict_Acc:dict = primary_feat(df_Acc, Acc=True, order='corr')
    corr_XY_Acc, corr_YZ_Acc, corr_ZX_Acc = corr_dict_Acc['corr_XY'], corr_dict_Acc['corr_YZ'], corr_dict_Acc['corr_ZX']

    # Feature extraction for AccMag
    # Extracting features : Using AccMag (1st order feature) : 3
    AccMag_mean = AccMag.mean()
    AccMag_std = AccMag.std()
    AccMag_sum = AccMag.sum()

    # Feat extract for AccMag_2_1 : 3
    AccMag_2_1_mean = AccMag_2_1.mean()
    AccMag_2_1_std = AccMag_2_1.std()
    AccMag_2_1_sum = AccMag_2_1.sum()

    # Extracting features : AccZ/AccMag - 1st order feature : 3
    AccZ_scaled_mean = AccZ_scaled.mean()
    AccZ_scaled_std = AccZ_scaled.std()
    AccZ_scaled_sum = AccZ_scaled.sum()
    # Extracting feat (mean, std, sum) : AccZ/AccMag_2_1 : 3
    AccZ_scaled_2_1_mean, AccZ_scaled_2_1_std, AccZ_scaled_2_1_sum = extract_feat_mss(AccZ_scaled_2_1)

    result_dict_Acc = {'corr_XY_Acc': corr_XY_Acc, 'corr_YZ_Acc': corr_YZ_Acc, 'corr_ZX_Acc': corr_ZX_Acc,
                       'AccMag_mean': AccMag_mean, 'AccMag_std': AccMag_std, 'AccMag_sum': AccMag_sum,
                       'AccMag_2_1_mean': AccMag_2_1_mean, 'AccMag_2_1_std': AccMag_2_1_std,
                       'AccMag_2_1_sum': AccMag_2_1_sum, 'AccZ_scaled_mean': AccZ_scaled_mean,
                       'AccZ_scaled_std': AccZ_scaled_std,
                       'AccZ_scaled_sum': AccZ_scaled_sum, 'AccZ_scaled_2_1_mean': AccZ_scaled_2_1_mean,
                       'AccZ_scaled_2_1_std': AccZ_scaled_2_1_std,
                       'AccZ_scaled_2_1_sum': AccZ_scaled_sum}

    gyro_dict = primary_feat(df_Gyro)
    # [0] is dict, [1] is a simple list of features. We choose dict
    gyro_dict = gyro_dict[0]
    # 1st order feat
    GyroMag = gyro_dict['First_features']['GyroMag']
    # 2nd order feat
    GyroMag_2_1 = gyro_dict['Second_features']['GyroMag_2_1']

    # Apply mean, std, sum for each nth-order feature : 6
    GyroMag_mean, GyroMag_std, GyroMag_sum = extract_feat_mss(GyroMag)
    GyroMag_2_1_mean, GyroMag_2_1_std, GyroMag_2_1_sum = extract_feat_mss(GyroMag_2_1)

    # Get Gyro corr coef : 3
    corr_dict_Gyro = primary_feat(df_Gyro, order='corr')
    corr_XY_Gyro, corr_YZ_Gyro, corr_ZX_Gyro = corr_dict_Gyro['corr_XY'], corr_dict_Gyro['corr_YZ'], corr_dict_Gyro[
        'corr_ZX']

    ### TOTAL: extracted 24 FEATURES for Accel, Gyro sensors, using 1st order feature & Second_order_1, with also computing
    ### correlation_coef for each two axis for each sensor_type.
    # Features of Gyro + Label
    result_dict_Gyro = {
        # Features of Gyro
        # corr of Gyro
        'corr_XY_Gyro': corr_XY_Gyro, 'corr_YZ_Gyro': corr_YZ_Gyro, 'corr_ZX_Gyro': corr_ZX_Gyro,
        # mean, std, sum for GyroMag(1st-order)
        'GyroMag_mean': GyroMag_mean, 'GyroMag_std': GyroMag_std, 'GyroMag_sum': GyroMag_sum,
        # mean ,std, sum for GyroMag_2_1 (2nd-order f)
        'GyroMag_2_1_mean': GyroMag_2_1_mean, 'GyroMag_2_1_std': GyroMag_2_1_std,
        'GyroMag_2_1_sum': GyroMag_2_1_sum,
        'activity': label  # not labelled yet
    }
    result_dict_Acc.update(result_dict_Gyro)
    result_dict = result_dict_Acc

    return result_dict


path = r'C:\Users\USER\Desktop\Machine_Learning\File\files\0Acc_[12.04] 12-11-14...12-11-30.txt'
df_ex, _ = utils.getDF_xyz(path)
v= primary_feat(df_ex, Acc=True, order=3)


