'''
액티비티 별로 나눈다.

'''


import Utils_ as utils
from preprocessing1 import get_df_raw
from preprocessing1 import primary_feat
from preprocessing1 import extract_feature_row
import Data_info

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


entire_data_list = []



def createDfForTraining(sensor_type: str, folder_dir: str, rel_path: str) -> list:

    if sensor_type not in Data_info.SENSOR_TYPE:
        raise Exception("sensor_type param must be in Data_info.py's SENSOR_TYPE")

    # print('main started')
    # sensor_type = 'Acc/AccZ'
    # # 파일들이 위치한 폴더 경로
    # folder_dir = r'C:\Users\USER\Desktop\Joljak\data\files'
    # rel_path = r'data/files/'

    # 각 ACC파일의 상대경로를 List에 모두 저장하여 Get하기.
    all_file_names = utils.getAllFileNames(folder_dir, sensor_type)


    # all_file_paths = [rel_path + accFilename for accFilename in all_file_names]
    all_file_paths = utils.convert2Paths(rel_path, all_file_names)

    time = 1 # second

    data_list_this_folder = []

    # For One File
    for filePath in all_file_paths:

        if all_file_paths.index(filePath) != 0:
            data_list_this_file.clear()
            # break;
        else: # 처음 iteration 이면
            data_list_this_file = []

        # read an entire file.
        df_entire, label, fs_int = get_df_raw(filePath, len(rel_path))
        # 이때 entire_df 는 이미 3초 전후 데이터가 빠져있다.

        winsize = fs_int * time

        remain_nrows = len(df_entire) % winsize
        # 꼬투리는 제거
        df_entire = df_entire[0:-(remain_nrows)]

        quotient = len(df_entire)//winsize

        # print('quotient =', quotient)

        # per one window block
        for i in range(quotient):
            # if i != 0:
            #     break

            df_window = df_entire[i*winsize: i*winsize + winsize]

            # print("df window:", df_window, sep='\n')

            primary_features, Corr = primary_feat(df_window, sensor_type, get_corr=True)

            acc_result_dict = extract_feature_row(primary_features, sensor_type, Corr)
            
            # 속성에 label 추가
            acc_result_dict['label'] = label


            data_list_this_file.append(acc_result_dict)

        data_list_this_folder.extend(data_list_this_file)

    return data_list_this_folder




            # print('acc_result_dict')
            # print(acc_result_dict)
            # if i == 0:
            #     print('i =', i)
            #     print(acc_result_dict)





    # data_list: list = []
    # dict_temp: dict = {}
    #
    # # generate 1 feature-row per 1 acc file.
    # for filePath in all_file_paths:
    #     # Extract features from files, gather them in a list as dictionaries.
    #     dict_temp: dict = preprocessing.extract_feature_row(filePath, rel_path_len=len(rel_path))
    #     data_list.append(dict_temp)
    #
    # # Dataframe으로 만들기
    # df_fin: pd.DataFrame = pd.DataFrame(data_list)
    #
    # # 타켓콜룸 정렬
    # df_fin: pd.DataFrame = df_fin[[c for c in df_fin if c is not 'activity']+['activity']]
    #
    # # Row 셔플링(Shuffling)
    # from sklearn.utils import shuffle
    # df_fin = shuffle(df_fin)
    # df_fin.reset_index(inplace=True, drop=True)
    #
    # # Save as a .csv file : CSV파일로  쓰기
    # df_fin.to_csv(r'C:\Users\USER\Desktop\Grad\GradProject\Generated_File\generated-file.csv', index=False)
    #
    # # separate features and labels
    # X = df_fin.drop('activity', 1) # Label 제거
    # y = np.array(df_fin.iloc[:, -1]).ravel()
    #
    # clf_lda = LinearDiscriminantAnalysis()
    # clf_qda = QuadraticDiscriminantAnalysis()
    #
    # from sklearn.model_selection import cross_val_score
    # cv_results_lda = cross_val_score(clf_lda, X, y, cv=5)
    # cv_results_qda = cross_val_score(clf_qda, X, y, cv=5)
    #
    # print('lda', cv_results_lda, np.mean(cv_results_lda))
    # print('qda', cv_results_qda, np.mean(cv_results_qda))
    #
    #
    #
    # # Do 5-fold cross-validation
    # # from sklearn.model_selection import cross
    #
    # # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True, stratify=True)
