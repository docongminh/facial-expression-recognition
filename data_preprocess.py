import numpy as np
import pandas as pd
from data_aug import data_augmentation
from sklearn.model_selection import train_test_split

def load_data(path):
    '''
    :param path:
    :return: dataframe dataset emotion 2013 kaggle
    '''
    return pd.read_csv(path)

def str2arr(data):

    '''
    :param data:
    :return: numpy array of images
    '''
    height = 48
    width = 48
    faces = []
    for point in data:
        face = [int(pixel) for pixel in point.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))

    return np.asarray(faces)


def extract_data(path):
    '''

    :param path: path to datasets
    :return: data training, data validation, data test
    '''
    datasets = load_data(path)
    usages = datasets.iloc[:, 2].tolist()

    count_usage = {}
    for usage in set(usages):
        count_usage[usage] = usages.count(usage)

    training_df = datasets[: count_usage['Training']]
    private_df = datasets[count_usage['Training']: count_usage['Training'] + count_usage['PublicTest']]
    public_df = datasets[count_usage['Training'] + count_usage['PublicTest']:]

    X_train = str2arr(training_df.iloc[:, 1].values)
    Y_train = training_df.iloc[:, 0].values
    X_val = str2arr(private_df.iloc[:, 1].values)
    Y_val = private_df.iloc[:, 0].values
    X_test = str2arr(public_df.iloc[:, 1].values)
    Y_test = public_df.iloc[:, 0].values

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def augmentation_preprocess(path):

    datasets = load_data(path)
    Disgust = datasets[datasets.emotion == 1].iloc[:, 1].values
    disgust_data = str2arr(Disgust)
    aug_images = data_augmentation(disgust_data)
    aug_remove_err = [im for im in aug_images if im.shape[0] == 48 and im.shape[1] == 48]
    cvt_np_arr = np.asarray(aug_remove_err)
    labels = np.ones(len(cvt_np_arr))

    #visulize data aug

    fig, axis = plt.subplots(6, 6, figsize=(12, 12))
    for i in range(6):
        for j in range(6):
            axis[i, j].imshow(cvt_np_arr[i], cmap="Greys")
            axis[i, j].axis("off")
    plt.show()

    # split train test validation
    aug_xtrain, aug_xtest, aug_ytrain, aug_ytest = train_test_split(cvt_np_arr, labels, test_size=0.2, random_state=0)

    xtrain, x_val, ytrain, y_val = train_test_split(aug_xtrain, aug_ytrain, test_size=0.2, random_state=0)

    return xtrain, x_val, ytrain, y_val, aug_xtest, aug_ytest



if __name__ == '__main__':

    dict_label = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6

    }

    print(extract_data(load_data('data/fer2013.csv')))



