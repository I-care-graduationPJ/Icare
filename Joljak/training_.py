
import Utils_ as utils
import preprocessing
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def main():

    # 파일들이 위치한 폴더 경로
    folder_dir = r'C:\Users\USER\Desktop\Joljak\data\files'
    rel_path = r'data/files'

    # 각 ACC파일의 상대경로를 List에 모두 저장하여 Get하기.
    allAccFileNames = utils.getAllFileNames(folder_dir)

    # allAccFilePaths = [rel_path + accFilename for accFilename in allAccFileNames]
    allAccFilePaths = utils.convert2Paths('files/', allAccFileNames)


    data_list: list = []
    dict_temp: dict = {}

    # generate 1 feature-row per 1 acc file.
    for filePath in allAccFilePaths:
        # Extract features from files, gather them in a list as dictionaries.
        dict_temp: dict = preprocessing.extract_feature_row(filePath, rel_path_len=len(rel_path))
        data_list.append(dict_temp)

    # Dataframe으로 만들기
    df_fin: pd.DataFrame = pd.DataFrame(data_list)

    # 타켓콜룸 정렬
    df_fin: pd.DataFrame = df_fin[[c for c in df_fin if c is not 'activity']+['activity']]

    # Row 셔플링(Shuffling)
    from sklearn.utils import shuffle
    df_fin = shuffle(df_fin)
    df_fin.reset_index(inplace=True, drop=True)
    df_final = df_fin

    # Save as a .csv file : CSV파일로  쓰기
    df_final.to_csv(r'C:\Users\USER\Desktop\Grad\GradProject\Generated_File\generated-file.csv', index=False)

    # separate features and labels
    X = df_final.drop('activity', 1) # Label 제거
    y = np.array(df_final.iloc[:, -1]).ravel()

    clf_lda = LinearDiscriminantAnalysis()
    clf_qda = QuadraticDiscriminantAnalysis()

    from sklearn.model_selection import cross_val_score
    cv_results_lda = cross_val_score(clf_lda, X, y, cv=5)
    cv_results_qda = cross_val_score(clf_qda, X, y, cv=5)

    print('lda', cv_results_lda, np.mean(cv_results_lda))
    print('qda', cv_results_qda, np.mean(cv_results_qda))



    # Do 5-fold cross-validation
    # from sklearn.model_selection import cross

    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=True, stratify=True)