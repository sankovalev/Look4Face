import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

#DATASET_PATH = '/home/velavok/ML/DEEPLOM/data/for_FACEEVOLVE/imgs' #папка с данными (каждая подпапка-отдельный класс)
#TRAIN_FOLDER = '/home/velavok/ML/DEEPLOM/data/for_FACEEVOLVE/imgs_train'
#TEST_FOLDER = '/home/velavok/ML/DEEPLOM/data/for_FACEEVOLVE/test'


#class_ratio = 0.05 #какая доля от всех классов будет использована для теста
#images_ratio = 0.05 #какая доля от фоток в одном классе будет использована для теста
#num_chunks = 9 #на сколько папок разбивать тренировочную выборку (чтобы не мучаться с памятью)

def make_train(DATASET_PATH, TRAIN_FOLDER, num_chunks = 9, remove_original = False):
    '''
    :param DATASET_PATH:
    :param TRAIN_FOLDER:
    :param num_chunks:
    :return:
    '''
    if not os.path.isdir(TRAIN_FOLDER):
        os.mkdir(TRAIN_FOLDER)

    classes = os.listdir(DATASET_PATH)
    N = len(classes)
    chunks = np.array_split(classes, num_chunks)
    for part_num in np.arange(len(chunks)):
        part_path = os.path.join(TRAIN_FOLDER, f'part_{str(part_num + 1)}')
        os.mkdir(part_path)
        for folder in tqdm(chunks[part_num]):
            src_path = os.path.join(DATASET_PATH, folder)  # откуда
            dest_path = os.path.join(TRAIN_FOLDER, part_path, folder)  # куда
            shutil.copytree(src_path, dest_path)
            if remove_original:
                os.rmdir(src_path)
        print(f'Chunk {str(part_num + 1)} of {str(len(chunks))} copied!')


def make_test(DATASET_PATH, TEST_FOLDER, class_ratio = 0.05, images_ratio = 0.05, TEST_CSV = 'test.csv'):
    '''
    :param DATASET_PATH:
    :param TEST_FOLDER:
    :param class_ratio:
    :param images_ratio:
    :param TEST_CSV:
    :return:
    '''
    if not os.path.isdir(TEST_FOLDER):
        os.mkdir(TEST_FOLDER)

    test_dict = dict()
    classes = os.listdir(DATASET_PATH)
    N = len(classes)
    test_classes = np.random.choice(classes, int(N * class_ratio))  # выбрали классы для теста
    # делаем тестовую выборку
    counter = 1
    for test_class in test_classes:
        images = os.listdir(os.path.join(DATASET_PATH, test_class))  # все картинки класса
        test_images = np.random.choice(images, int(len(images) * images_ratio),
                                       replace=False)  # выбрали картинки для теста
        for img in test_images:
            img_src = os.path.join(DATASET_PATH, test_class, img)
            filename = f"face_{counter}.{img.split('.')[-1]}"
            img_dest = os.path.join(TEST_FOLDER, filename)
            shutil.copy(img_src, img_dest)
            os.remove(img_src)
            test_dict[filename] = test_class
            counter += 1
    test_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=['id']).reset_index()
    test_df.to_csv(TEST_CSV, header=False, index=False)
    
    
def make_train_new(TRAIN_FOLDER, TRAIN_NEW_FOLDER, class_ratio = 1, images_ratio = 0.05):
    '''
    Если уже есть тренировочная, разбитая на части
    :param TRAIN_FOLDER: тут папки part_1, part_2, ... с тренировочной выборкой
    :param TRAIN_NEW_FOLDER: тут будет такая же структура, только фоток меньше
    :param class_ratio: какую долю классов брать
    :param images_ratio: какую долю изображений брать
    :return:
    '''
    if not os.path.isdir(TRAIN_NEW_FOLDER):
        os.mkdir(TRAIN_NEW_FOLDER)
    
    for part in os.listdir(TRAIN_FOLDER):
        if not os.path.isdir(os.path.join(TRAIN_NEW_FOLDER, part)):
            os.mkdir(os.path.join(TRAIN_NEW_FOLDER, part))
        classes = os.listdir(os.path.join(TRAIN_FOLDER, part))
        N = len(classes)
        train_classes = np.random.choice(classes, int(N * class_ratio), replace=False)  # выбрали (все) классы
        for train_class in train_classes:
            os.mkdir(os.path.join(TRAIN_NEW_FOLDER, part, train_class)) #создать папку
            images = os.listdir(os.path.join(TRAIN_FOLDER, part, train_class))  # все картинки класса
            train_images = np.random.choice(images, int(len(images) * images_ratio),
                                           replace=False)  # выбрали картинки для теста
            for img in train_images:
                img_src = os.path.join(TRAIN_FOLDER, part, train_class, img) #откуда
                img_dest = os.path.join(TRAIN_NEW_FOLDER, part, train_class, img) #куда
                shutil.copy(img_src, img_dest)



