# class data_info():

FOLDER_PATH = r'data/files/'
LABEL_INFO = {0:'writing', 1:'lifting', 2:'walking', 3:'sitting'}
SENSOR_TYPE = ['Acc/AccZ', 'Gyro']

def get_sensor_type(ch):
    if ch in ['a', 'A', 0, 'Acc', 'acc']:
        return SENSOR_TYPE[0]

    elif ch in ['g', 'G', 1]:
        return SENSOR_TYPE[1]



