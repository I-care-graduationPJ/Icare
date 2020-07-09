import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from os.path import dirname, join


# Read a Local File (csv)
def getDataFrameFromLocalFile() -> pd.DataFrame:

    f = open(join(dirname(__file__), 'sample_data2.csv'))
    df = pd.read_csv(f)
    return df



# train with the Dataframe produced from readLocalFile() function
def get_trained_lda_clf(df: pd.DataFrame) -> LinearDiscriminantAnalysis :
    lda = LinearDiscriminantAnalysis()
    # qda = QuadraticDiscriminantAnalysis()

    # separate features and labels
    X = df.drop('activity', 1) # Label 제거
    y = np.array(df.iloc[:, -1]).ravel()

    lda.fit(X, y)

    return lda

    # from sklearn.model_selection import cross_val_score
    # cv_results_lda = cross_val_score(lda, X, y, cv=5)
    # cv_results_qda = cross_val_score(qda, X, y, cv=5)

    # print('lda', cv_results_lda, np.mean(cv_results_lda))
    # print('qda', cv_results_qda, np.mean(cv_results_qda))

def predict(lda:LinearDiscriminantAnalysis, X:list):
    y = lda.predict(X)


PATH = R'C:\Users\USER\Desktop\Joljak\data\Generated_File\Activity-recognition.csv'
df = pd.read_csv(PATH)
lda = get_trained_lda_clf(df)
X = df.drop("activity", 1)
input = np.array(X.iloc[0, :]).reshape(1, -1)
res = lda.predict(input)

# return type: numpy.ndarray
print(res, type(res))
