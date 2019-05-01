# Helper function for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from functools import partial
# import pickle
# pickle.load = partial(pickle.load, encoding="latin1")
# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")


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
# TODO: передавать размер картинки как параметр
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def extract_one_embedding(image, backbone, model_root,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    '''
    :param img_root: путь к изображению
    :param backbone: тело сетки
    :param model_root: папка с весами модели
    :param device: cpu или gpu
    :param tta: надо ли брать зеркальное отображение лица
    :return: вектор признаков
    '''

    image = image.convert('RGB')
    image = transform(image)
    ccropped = np.reshape(image, [1, 3, 112, 112])
    flipped = hflip(image)
    flipped = np.reshape(flipped, [1, 3, 112, 112])

    # load backbone from a checkpoint
    model = torch.load(model_root) #, map_location=lambda storage, loc: storage, pickle_module=pickle)
    backbone.load_state_dict(model)
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
