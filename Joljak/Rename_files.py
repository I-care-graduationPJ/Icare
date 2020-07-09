import Utils_ as utils


# To find a list of fileNames at once.
def getFileNames(path, sensorType, date_str=None, time_start=None, time_end=None):
    '''
    Returns a list of filepaths specified by date and time
    The first number indicates the activity. (ex-0:write, 1:lifting-hands, etc..)

    :param sensorType: string - 'Acc', 'Gyro'
    :param date_str: date - '11.25'
    :param time_start: '10-11'
    :param time_end:
    :return: a list of file paths
    '''


    # rel_path = 'files/'
    # path = r'C:\Users\USER\Desktop\Machine_Learning\File\files'

    from os import listdir
    from os.path import isfile, join
    # extract file names
    onlyFiles = [f for f in listdir(path) if isfile(join(path, f))]
    # print("Inside [getFilesList] function:")
    # print(onlyFiles)
    sensorType = sensorType + '_'

    if date_str!=None:
        sensorType = sensorType + "[" + date_str + "]"

        if time_start!=None:
            sensorType = sensorType + " " + time_start

    vals = [str for str in onlyFiles if str.startswith(sensorType)]
    final_val = vals

    # filePaths = [rel_path + filename for filename in onlyFiles]
    return final_val

# rename files
def rename_file(parent_path:str, old_name:str, new_name:str)->None:
    '''
    Rename a file, given its path
    :param parent_path: Must add 'r' before path-string. Must end with backslash.
    :param old_name: old_file Name
    :param new_name: new)fileName
    :return: None
    '''
    import os
    os.rename(parent_path + old_name, parent_path + new_name)
    print("changed from", old_name, 'to', new_name)

path = r'C:\Users\USER\Desktop\Machine_Learning\File\liftingHands\files'

def change_file_names(path, fileName_list, activity):
    '''

    :param fileName_list: list of file names
    :param activity: must be an integer from 0-6
    :return:
    '''

    if activity not in list(range(7)):
        print('activity param error - must be 0~6')
        return


    print('original fileNames')
    for i in fileName_list:
        print(i)
    print(len(fileName_list))

    # rename them - setting label: lifting=1, writing=0
    for fileName in fileName_list:
        rename_file(path+'\\', fileName, str(activity) + fileName)
    print('Renaming successfully done')

# change_file_names('Gyro')

# # Change file names -> add 0 for acc
# writing_10sec_fileName = getFileNames(path=r'C:\Users\USER\Desktop\Machine_Learning\File\files', sensorType='Acc', date_str='12.16')
# for i in writing_10sec_fileName:
#     print(i)
# change_file_names( r'C:\Users\USER\Desktop\Machine_Learning\File\files', writing_10sec_fileName, 0)
#
# # Change file names -> add 0 for gyro
# writing_10sec_fileName_g = getFileNames(path=r'C:\Users\USER\Desktop\Machine_Learning\File\files', sensorType='Gyro', date_str='12.16')
# for i in writing_10sec_fileName_g:
#     print(i)
# change_file_names( r'C:\Users\USER\Desktop\Machine_Learning\File\files', writing_10sec_fileName_g, 0)

a=utils.getAllFileNames(r'C:\Users\USER\Desktop\Machine_Learning\File\files')
print(len(a))
for i in a:
    print(i)