from training_1 import createDfForTraining
from Data_info import get_sensor_type
from Data_info import LABEL_INFO
import Utils_ as utils
from preprocessing1 import get_df_raw
from preprocessing1 import primary_feat
from preprocessing1 import extract_feature_row
import Data_info
import preprocessing1 as pre
import pandas as pd
import numpy as np



folder_dir = r'C:\Users\USER\Desktop\Joljak\data\files2'
file_rel_path = r'data/files2/'
sensor_type = get_sensor_type('g')


# Create a list of rows(dictionaries)
# createDfForTraining(get_sensor_type('a'), folder_dir, rel_path)


# 여기서, 모든 acc, gyro 파일을 뽑지말고, 이 중에서 60초 이상인 유의미한 데이터를 뽑아낸다.
all_gyro_file_names = utils.getAllFileNames(folder_dir, get_sensor_type('g'))
all_acc_file_names = utils.getAllFileNames(folder_dir, get_sensor_type('a'))

all_gyro_file_paths = utils.convert2Paths(file_rel_path, all_gyro_file_names)
all_acc_file_paths = utils.convert2Paths(file_rel_path, all_acc_file_names)

# 삭제 금지 - 파일들 중, 60초 이상인 파일들만 뽑아내는 코드.
files_long_enough_acc = []
files_long_enough_gyro = []
duration_acc = []
duration_gyro = []

for file_rel_path1 in all_acc_file_paths:
    period = utils.getDF_xyz(file_path_rel=file_rel_path1, rel_path=file_rel_path)[3]
    if period < 6:
        continue
    files_long_enough_acc.append(file_rel_path1)
    duration_acc.append(period)

for rel_path1 in all_gyro_file_paths:
    period = utils.getDF_xyz(rel_path1, file_rel_path)[3]
    if period < 6:
        continue
    files_long_enough_gyro.append(rel_path1)
    duration_gyro.append(period)

print('duration acc:', duration_acc)
print('duration gyro:', duration_gyro)


entire_rows = []

do_filter:bool = None

for i in range(len(files_long_enough_acc)):
    if i == 0:
        this_file_data = []
    else:
        this_file_data.clear()


    # select this file
    a_path = files_long_enough_acc[i]
    g_path = files_long_enough_gyro[i]

    df_a, label_this_file, fs_a, period_a = utils.getDF_xyz(a_path, file_rel_path)
    df_g, _, fs_g, period_g = utils.getDF_xyz(g_path, file_rel_path)

    # # condition check for fs(sampling rate) = Wn
    # nyq = 0.5 * fs_a
    # normal_cutoff_a = 0.5 / nyq
    # if normal_cutoff_a < 0 or normal_cutoff_a > 1:
    #     do_filter = False
    # # nyq = 0.5 * fs_g
    # # normal_cutoff_g = cutoff / nyq


    cut_a = int(3*fs_a)
    cut_g = int(3*fs_g)

    # # 전처리: 4초
    df_a = df_a[cut_a:-cut_a].reset_index(drop=True)
    df_g = df_g[cut_g:-cut_g].reset_index(drop=True)

    len_a = len(df_a)
    len_g = len(df_g)
    last_index_a = len_a - 1
    last_index_g = len_g - 1

    time=1 #second

    # winsize for each sensor
    winsize_a:int = int(time*fs_a)
    winsize_g:int = int(time*fs_g)

    remain_a = len(df_a) % winsize_a
    remain_g = len(df_g) % winsize_g

    print('remain_a:', remain_a, "remain_g:", remain_g)

    # integer
    remain_a = int(remain_a) +1
    remain_g = int(remain_g) +1

    # 애초에 60초 이상인 데이터들을 뽑는다.

    i = 0
    while((i*winsize_a + winsize_a <= last_index_a)  and (i*winsize_a < last_index_a)
          and (i*winsize_g + winsize_g <= last_index_g) and (i*winsize_g < last_index_g)):

        start_a = i*winsize_a
        end_a = i*winsize_a + winsize_a

        start_g = i*winsize_g
        end_g  = i*winsize_g + winsize_g

        # Caution !
        # Digital filter critical frequencies must be 0 < Wn < 1

        # nyq = 0.5 * fs
        # normal_cutoff = cutoff / nyq


        pri_feat_a, corr_a = primary_feat(df_a[start_a:end_a], get_sensor_type('a'))
        pri_feat_g, corr_g = primary_feat(df_g[start_g: end_g], get_sensor_type('g'))

        # each is a dict
        one_row_a = extract_feature_row(pri_feat_a, get_sensor_type('a'), corr_a)
        one_row_g = extract_feature_row(pri_feat_g, get_sensor_type('g'), corr_g)

        # final dict containing all the features
        one_row_a.update(one_row_g)
        one_row_all = one_row_a
        # add 'activity' to the dict : same activty for both 'a' and 'g' - convert it to string
        # 수정했음
        # one_row_all['activity'] = LABEL_INFO[label_this_file]
        one_row_all['activity'] = label_this_file

        this_file_data.append(one_row_all)

        i+=1

    entire_rows.extend(this_file_data)

# Dataframe 화
df = pd.DataFrame(entire_rows)

# 타켓콜룸 정렬
df: pd.DataFrame = df[[c for c in df if c is not 'activity']+['activity']]

# Row 셔플링(Shuffling)
from sklearn.utils import shuffle
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

# Save as a .csv file : CSV파일로  쓰기
df.to_csv(r'C:\Users\USER\Desktop\Joljak\data\Generated_File\result0.csv', index=False)

# separate features and labels
# X = df.drop('activity', 1) # Label 제거
# y = np.array(df.iloc[:, -1]).ravel()

# clf_lda = LinearDiscriminantAnalysis()
# clf_qda = QuadraticDiscriminantAnalysis()




# 삭제 금지 - 파일들 중, 60초 이상인 파일들만 뽑아내는 코드.
# files_long_enough = []
# duration = []
#
# for rel_path in all_acc_file_paths:
#     period = utils.getDF_xyz(rel_path)[2]
#     # if period < 60:
#     #     continue
#     # files_long_enough.append(rel_path)
#     duration.append(period)
#
# print(duration)
# # print(files_long_enough)



# def create_feat_for_whole_folder(folder_dir:str, rel_path:str)->dict:

# list of dictionaries
# AccFeatures:list = createDfForTraining(get_sensor_type('a'), folder_dir, rel_path)
# GyroFeatures:list = createDfForTraining(get_sensor_type('g'), folder_dir, rel_path)

# # 어차피 Acc-Gyro 한 쌍으로 생성되기 때문에, list에 추가되는 dict의 순서도 같다.
# # since both list are of same order
# for i in range(len(AccFeatures)):
#     AccFeatures[i].update(GyroFeatures[i])
# complete_features = AccFeatures
#
# return complete_features

# print(create_feat_for_whole_folder(folder_dir, rel_path))

# acc = all_acc_file_paths[0]
# gyro = all_gyro_file_paths[0]
#
# info_acc = utils.getDF_xyz(acc)
# df_acc = info_acc[0]
# fs_acc = info_acc[1]
# period_acc = info_acc[2]
#
# info_gyro = utils.getDF_xyz(gyro)
# df_gyro = info_gyro[0]
# fs_gyro = info_gyro[1]
# period_gyro = info_gyro[2]

