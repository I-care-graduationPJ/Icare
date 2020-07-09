from builtins import tuple
from typing import Any, Union

# activity 설명:
# 0: writing
# 1: lifting

import numpy as np
import pandas as pd
import Utils_ as utils
from lowPassFilter import do_lpf
from lowPassFilter import butter_lowpass_filter
from Utils_2 import get_norm
from Utils_2 import extract_feat_mss
from Utils_2 import get_corr
import bandPassFilter


# Apply low-pass-filter
def do_lpf(data_arr: np.ndarray, fs: float, order: int = 1, cutoff: float = 0.5) -> np.ndarray:
    """
    :param data_arr: data-array to process
    :param fs: sampling rate
    :param order: n
    :param cutoff: float
    :return: Filtered ndarray
    """
    # Apply low pass filter on VM
    filtered_data = butter_lowpass_filter(data=data_arr, cutoff=cutoff, fs=fs, order=order)
    return filtered_data


def get_df_raw(file_path_rel: str, rel_path_len: int)->tuple:
    '''
    Produce a pd.DataFrame & dict(its info) from the input textFile.
    Extra processing: delete 1second(sample rate) from the start and the end for making the data quality high

    :param file_path: relative path to file
    :param rel_path_len: length of relative file path
    :return:
        tuple of (dataframe, label(int))

    '''

    # 파일 이름으로 부터 Label값 얻기.
    starting_index:int = rel_path_len
    label: int = int(file_path_rel[starting_index])

    # # 수정
    # # only replace this part of the string.
    # file_path_Gyro = file_path_rel.replace('Acc_', 'Gyro_')
        
    df, fs, period = utils.getDF_xyz(file_path_rel, getDfInfo=True)

    fs_int = int(fs)

    # 데이터 정제 (3초 앞뒤로 빼줌)
    df = df[3*fs_int:-3*fs_int]

    # fs = sample_rate = df_info['sample rate']
    # period:float = df_info['period']
    # # Data Processing
    # df = df[fs:-fs]

    return df, label, fs_int

# function to check frequencies
def fs_good(fs, cutoff:float) -> bool:
    nyq = fs*cutoff
    normal_cutoff = cutoff / nyq
    if normal_cutoff < 0 or normal_cutoff > 1:
        return True
    else:
        return False

# Extract primary features of a given raw-data dataframe
# If it is Acc, set Acc=True
def primary_feat(df: pd.DataFrame, sensor_type :str, get_corr=True) -> tuple:
    '''

    :param df: Dataframe to process
    :param sensor_type: 'Acc' or 'Gyro'
    :param order: 'corr' : exrract correlation vals(3 values) from filtered data X,Y,Z
    :return: dictionary of primary-features
    '''


    # 타입 체크
    if type(df) != pd.DataFrame:
        print('param error: df must be a pd.Dataframe - 70줄')
        print('preprocessing.primary_feat function ends')
        return

    if type(sensor_type) != str:
        raise Exception("sensor_type 다시 입력 - String 값")
        return
    elif sensor_type not in ['Acc/AccZ', 'Gyro']:
        raise Exception('에러 @@@ --- Acc/Gyro 중 하나 여야 함')
        return


    # Sample Rate, Duration 추출
    period = (df.iloc[-1, 0] - df.iloc[0, 0])/10**9
    fs: float = len(df) / period  # sample rate, Hz ('fs' number of data per second)
    nyq = 0.5 * fs


    # separate data
    X_time: pd.Series = df['timestamp']
    X_data: pd.Series = df['x_data']
    Y_data: pd.Series = df['y_data']
    Z_data: pd.Series = df['z_data']

    X_data_fil = None
    Y_data_fil = None
    Z_data_fil = None


    if sensor_type == 'Acc/AccZ':
        cutoff = 0.4
        order = 1
        normal_cutoff = cutoff / nyq
        if normal_cutoff < 0 or normal_cutoff > 1:
            X_data_fil = X_data
            Y_data_fil = Y_data
            Z_data_fil = Z_data
        else:
            # Apply Low-pass-Filter
            X_data_fil: np.array = do_lpf(X_data, fs, order, cutoff)
            Y_data_fil = do_lpf(Y_data, fs, order, cutoff)
            Z_data_fil = do_lpf(Z_data, fs, order, cutoff)

    elif sensor_type == 'Gyro':
        # cutoff = 0.1
        # order = 1
        X_data_fil = X_data
        Y_data_fil = Y_data
        Z_data_fil = Z_data
    else:
        raise Exception("@@@@ error in primary_feat: paramater - sensor_type:str")


    # # Apply Low-pass-Filter
    # X_data_fil: np.array = do_lpf(X_data, fs, order, cutoff)
    # Y_data_fil = do_lpf(Y_data, fs, order, cutoff)
    # Z_data_fil = do_lpf(Z_data, fs, order, cutoff)

    Features = {}


    # Apply lpf to original X, Y, Z data - 0.5 cutoff, 1st order - filtered data: np array
    if sensor_type == 'Acc/AccZ':

        # Get Norm
        VecMag:list = utils.get_norm(X_data_fil, Y_data_fil, Z_data_fil)

        # print('VecMag:', VecMag)

        # Get AccZ/AccMag
        AccZ:np.ndarray = Z_data_fil / VecMag
        # print('Accz:', AccZ)

        ### 2nd order feature(1)
        # apply lpf once again

        # 수정됨 - low pass filter 한번더 하는건데, 취소함 -> fs가 낮아서 오류뜸 (sampling rate - too low)
        # if fs_good(fs, 0.4) == True:
        #     VecMag_2ndFeature_1 = do_lpf(VecMag, fs)  ##### fianl data
        # else:
        VecMag_2ndFeature_1 = VecMag

        # 수정됨 - low pass filter 한번 더 하는거 없앰. -> fs가 낮아서 오류뜸
        # AccZ_2ndFeature_1 = do_lpf(AccZ, fs)
        AccZ_2ndFeature_1 = AccZ

        # rename data to Acc
        AccMag = VecMag
        AccMag_2ndFeature_1 = VecMag_2ndFeature_1

        # 수정 - key값 수정
        # dict form
        Features.update({
            # 'sensor_type': 'Acc/AccZ',
            'first_features': {'AccMag': AccMag, 'AccZ': AccZ},
            'second_features': {'AccMag_2_1': AccMag_2ndFeature_1, 'AccZ_2_1': AccZ_2ndFeature_1}}
        )



    elif sensor_type == 'Gyro':

        # Get Norm
        VecMag = utils.get_norm(X_data_fil, Y_data_fil, Z_data_fil)
        # [ 수정 ]
        # VecMag_2ndFeature_1 = do_lpf(VecMag, fs)  ##### fianl data
        VecMag_2ndFeature_1 = VecMag

        # rename data to 'Gyro'
        GyroMag = VecMag
        GyroMag_2_1 = VecMag_2ndFeature_1

        # dict form
        Features.update({
            # 'sensor_type': 'Gyro',
            'first_features': {'GyroMag': GyroMag},
            'second_features': {'GyroMag_2_1': GyroMag_2_1}
        })
    else:
        raise Exception


    corr = {}
    # 수정
    # 메소드로 변환하자.
    # Find correlation values, and return them as dict (the funciton ends)
    if get_corr == True:
        # params: two variables that we want to find the correlation coefficient
        corr_XY = np.corrcoef(X_data_fil, Y_data_fil)[0, 1]
        corr_YZ = np.corrcoef(Y_data_fil, Z_data_fil)[0, 1]
        corr_ZX = np.corrcoef(Z_data_fil, X_data_fil)[0, 1]

        corr.update({'corr_XY': corr_XY, 'corr_YZ': corr_YZ, 'corr_ZX': corr_ZX})

    return Features, corr



    ### 2nd order feature(2) - Moving variance
    ### 1. 2nd order bandpass(0.15-0.20Hz),
    ### 2. compute the square of the individual samples
    ### 3. 2nd order bandpass(0.15-0.20Hz)
    # apply bandpass filter: 0.15~0.2Hz, 2nd-order
    # def do_second_2():
    #     norm_bpf2 = bandPassFilter.butter_bandpass_filter(VecMag, 0.15, 0.2, fs, 2)
    #     # compute the square of individual samples
    #     norm_bpf_squared = norm_bpf2 ** 2
    #     # Averaging the result with a low pass filter: 0.5HZ cuttoff
    #     norm_bpf_squared_avged = do_lpf(norm_bpf_squared)
    #
    #     # plt.plot(X_time, norm_bpf_squared, label='just doubled')
    # # do_second_2()
    # ### feature3: Movring RMS
    # def do_second3():
    #     # bpf, 2nd-order(0.1-2Hz), compute square of indivi sample,
    #     # apply lpf(0.5cut), compute square-root
    #     norm_bpf3 = bandPassFilter.butter_bandpass_filter(VecMag, 0.1, 0.2, fs, 2)
    #     # square individual
    #     norm_bpf3_squared = norm_bpf3 ** 2
    #     # apply lpf
    #     norm_bpf3_squared_lpf = do_lpf(norm_bpf3_squared)
    #
    #     # Do sqrt on each value
    #     norm3_final = np.sqrt(norm_bpf3_squared_lpf)
    #     secondF_3 = norm3_final




# 이미 레이블링된 파일에 대한 경로 제공한다.
# 상대경로: 이 .py파일 입장에서의 경로
# Caution: This uses preprocessing.primary_feat (primary features like norms, filtered norms, etc..)
# Gyro도 자동으로 path 제공

def extract_feature_row(primary_feat_info:dict, sensor_type:str, corr_dict:dict, win_size: object = '2s') -> dict:
    '''

    :param primary_feat_info: dict from primary_feat
    :param sensor_type: string value
    :param corr_dict: correlation dict from primary_feat
    :param win_size:
    :return:
        dict (for one row feature)
    '''
    # file_path_Acc = 'files/1Acc_[10.31] 10-20-16...10-20-32.txt'
    # file_path_Gyro = 'files/1Gyro_[10.31] 10-20-16...10-20-32.txt'


    # # Dataframe 생성 (for this filepath), df_info : 해당 파일의 samplingRate, period 정보
    # df, df_info = utils.getDF_xyz(file_path_Acc)


    # # For Gyro: get DF from textfile
    # df_Gyro, df_Gyro_info = utils.getDF_xyz(file_path_Gyro)

    # # 1초에 몇개의 instances?
    # fs_gyro = int(df_Gyro_info['fs'])
    # df_Gyro = df_Gyro[fs_gyro:-fs_gyro]

    # Primary Feature Extraction
    # primary_feat_info:dict = primary_feat(df)

    features = primary_feat_info # input (a dictionary)

    result_dict={}

    # if features['sensor_type'] == 'Acc/AccZ':
    if sensor_type == 'Acc/AccZ':
        AccMag = np.array(features['first_features']['AccMag'])
        AccMag_2_1 = np.array(features['second_features']['AccMag_2_1'])

        AccZ = np.array(features['first_features']['AccZ'])
        AccZ_2_1 = np.array(features['second_features']['AccZ_2_1'])

        # Correlation, using filtered Acc X,Y,Z data : 3
        # corr_dict_Acc:dict = primary_feat(df, Acc=True, order='corr')
        # corr_dict_Acc = get_corr(df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])
        corr_XY_Acc, corr_YZ_Acc, corr_ZX_Acc = corr_dict['corr_XY'], corr_dict['corr_YZ'], corr_dict['corr_ZX']

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
        AccZ_scaled_mean = AccZ.mean()
        AccZ_scaled_std = AccZ.std()
        AccZ_scaled_sum = AccZ.sum()

        # Extracting feat (mean, std, sum) : AccZ/AccMag_2_1 : 3
        AccZ_scaled_2_1_mean, AccZ_scaled_2_1_std, AccZ_scaled_2_1_sum = extract_feat_mss(AccZ_2_1)

        result_dict.update({'corr_XY_Acc': corr_XY_Acc, 'corr_YZ_Acc': corr_YZ_Acc, 'corr_ZX_Acc': corr_ZX_Acc,
                           'AccMag_mean': AccMag_mean, 'AccMag_std': AccMag_std, 'AccMag_sum': AccMag_sum,
                           'AccMag_2_1_mean': AccMag_2_1_mean, 'AccMag_2_1_std': AccMag_2_1_std, 'AccMag_2_1_sum': AccMag_2_1_sum,

                           'AccZ_mean': AccZ_scaled_mean,
                           'AccZ_std': AccZ_scaled_std,
                           'AccZ_sum': AccZ_scaled_sum,

                           'AccZ_2_1_mean': AccZ_scaled_2_1_mean,
                           'AccZ_2_1_std': AccZ_scaled_2_1_std,
                           'AccZ_2_1_sum': AccZ_scaled_sum}
                           )


    elif sensor_type is 'Gyro':

        # 1st order feat
        GyroMag = features['first_features']['GyroMag']
        # 2nd order feat
        GyroMag_2_1 = features['second_features']['GyroMag_2_1']

        # Apply mean, std, sum for each nth-order feature : 6
        GyroMag_mean, GyroMag_std, GyroMag_sum = extract_feat_mss(GyroMag)
        GyroMag_2_1_mean, GyroMag_2_1_std, GyroMag_2_1_sum = extract_feat_mss(GyroMag_2_1)

        # Get Gyro corr coef : 3
        # corr_dict_Gyro = primary_feat(df_Gyro, order='corr')
        corr_XY_Gyro, corr_YZ_Gyro, corr_ZX_Gyro = corr_dict['corr_XY'], corr_dict['corr_YZ'], corr_dict['corr_ZX']

        ### TOTAL: extracted 24 FEATURES for Accel, Gyro sensors, using 1st order feature & Second_order_1, with also computing
        ### correlation_coef for each two axis for each sensor_type.
        # Features of Gyro + Label
        result_dict.update({
            # Features of Gyro
            # corr of Gyro
            'corr_XY_Gyro': corr_XY_Gyro, 'corr_YZ_Gyro': corr_YZ_Gyro, 'corr_ZX_Gyro': corr_ZX_Gyro,
            # mean, std, sum for GyroMag(1st-order)
            'GyroMag_mean': GyroMag_mean, 'GyroMag_std': GyroMag_std, 'GyroMag_sum': GyroMag_sum,
            # mean ,std, sum for GyroMag_2_1 (2nd-order f)
            'GyroMag_2_1_mean': GyroMag_2_1_mean, 'GyroMag_2_1_std': GyroMag_2_1_std,
            'GyroMag_2_1_sum': GyroMag_2_1_sum,
            # 'activity': label  # not labelled yet
        })
    # result_dict_Acc.update(result_dict_Gyro)
    # result_dict = result_dict_Acc

    return result_dict


# path = r'C:\Users\USER\Desktop\Machine_Learning\File\files\0Acc_[12.04] 12-11-14...12-11-30.txt'
# df_ex, _ = utils.getDF_xyz(path)
# v= primary_feat(df_ex, Acc=True, order=3)


