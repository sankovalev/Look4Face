# Helper function for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from albumentations import Compose, Normalize, HorizontalFlip, Resize
# from albumentations.pytorch import ToTensor
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        
        images_list = []
        targets_list = []
        for folder in os.listdir(data_root):
            for image in os.listdir(os.path.join(data_root, folder)):
                images_list.append(os.path.join(data_root, folder, image))
                targets_list.append(folder)
        self.images_list = images_list
        self.targets_list = targets_list

    def __getitem__(self, index):
        image = Image.open(self.images_list[index])
        image = image.convert('RGB')
        target = self.targets_list[index]
    
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.images_list)


def l2_norm(input, axis = 1):
    '''
    L2-норма матрицы, т.е. евклидова норма построчно (корень из суммы квадратов элементов строки)
    :param input: tensor Nx512
    :param axis: 1-построчно, 0-постолбцово
    :return: пронормированный тензор (каждую строку поделили на ее норму, все значения теперь из [0;1])
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

# преобразования для зеркального изображения
hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

# преобразования для оригинального изображения
# TODO: get Resize parameters from function
# TODO: albumentations :))
# TODO: передавать размер картинки как параметр
transform = transforms.Compose([
            transforms.Resize([int(128 * 112 / 112), int(128 * 112 / 112)]),  # smaller side resized
            transforms.CenterCrop([112, 112]),
    #         Resize(input_size[0], input_size[1]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def extract_many_embeddings(data_root, backbone, model_root, input_size = [112, 112], rgb_mean = [0.5, 0.5, 0.5],
                    rgb_std = [0.5, 0.5, 0.5], embedding_size = 512, batch_size = 512,
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    '''
    Извлекает признаки из тренировочной выборки, где каждая папка - отдельный класс
    :param data_root: папка с папками/классами
    :param backbone: директория с сеткой
    :param model_root: директория с весами модели
    :param input_size: размер входного изображения
    :param rgb_mean: среднее
    :param rgb_std: стандартное отклонение
    :param embedding_size: количество фичей
    :param batch_size: размер батча
    :param device: cpu или gpu
    :param tta: надо ли брать зеркальное отображение лица
    :return: признаки, список путей к картинкам и их классы
    '''
    assert (os.path.exists(data_root))
    print('Testing Data Root:', data_root)
    assert (os.path.exists(model_root))
    print('Backbone Model Root:', model_root)

    dataset = TrainDataset(data_root, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 8)
    NUM_IMAGES = len(loader.dataset)
    print("Number of images: {}".format(NUM_IMAGES))

    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode
    idx = 0
    features = np.zeros([NUM_IMAGES, embedding_size])
    
    with torch.no_grad():
        for batch, _ in tqdm(loader):
            batch_len = len(batch)
            if tta:
                fliped = hflip_batch(batch)
                #берем вектор лица + вектор лица в зеркальном отображении
                emb_batch = backbone(batch.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                features[idx:idx + batch_len] = l2_norm(emb_batch)
            else:
                features[idx:idx + batch_len] = l2_norm(backbone(batch.to(device))).cpu()
            idx += batch_len

    return features, dataset.images_list, dataset.targets_list


def extract_one_embedding(img_root, backbone, model_root,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    '''
    :param img_root: путь к изображению
    :param backbone: тело сетки
    :param model_root: папка с весами модели
    :param device: cpu или gpu
    :param tta: надо ли брать зеркальное отображение лица
    :return: вектор признаков
    '''
    assert (os.path.exists(img_root))
    #print('Testing Data Root:', img_root)
    assert (os.path.exists(model_root))
    #print('Backbone Model Root:', model_root)

    image = Image.open(img_root)
    image = image.convert('RGB')
    image = transform(image)
    ccropped = np.reshape(image, [1, 3, 112, 112])
    flipped = hflip(image)
    flipped = np.reshape(flipped, [1, 3, 112, 112])

    # load backbone from a checkpoint
    #print("Loading Backbone Checkpoint '{}'".format(model_root))
    backbone.load_state_dict(torch.load(model_root))
    backbone.to(device)

    # extract features
    backbone.eval()  # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())

    return features
