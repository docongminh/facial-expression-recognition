from data_preprocess import load_data, augmentation_preprocess, extract_data
from setup_model import Model

import numpy as np

def main(path):
    '''

    :param path: path to datasets
    :return:
    '''
    X_train, Y_train, X_val, Y_val, X_test, Y_test = extract_data(path)
    xtrain, x_val, ytrain, y_val, aug_xtest, aug_ytest = augmentation_preprocess(path)

    new_X_train = np.concatenate((X_train, xtrain), axis=0)
    new_X_val = np.concatenate((X_val, x_val), axis=0)
    new_X_test = np.concatenate((X_test, aug_xtest), axis=0)

    print("Shape new_X_train: ", new_X_train.shape)
    print("Shape new_X_val: ", new_X_val.shape)
    print("Shape new_X_test: ", new_X_test.shape)

    new_Y_train = np.concatenate((Y_train.astype('int8'), ytrain.astype('int8')), axis=0)
    new_Y_val = np.concatenate((Y_val.astype('int8'), y_val.astype('int8')), axis=0)
    new_Y_test = np.concatenate((Y_test.astype('int8'), aug_ytest.astype('int8')), axis=0)
    print("Shape new_Y_train: ", new_Y_train.shape)
    print("Shape new_Y_val: ", new_Y_val.shape)
    print("Shape new_Y_test: ", new_Y_test.shape)



if __name__ == '__main__':
    path_data = 'data/fer2013.csv'
    main(path_data)


