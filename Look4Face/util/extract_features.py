# Functions for extracting features from pre-trained models
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from functools import partial


def l2_norm(input, axis = 1):
    """L2-norm of matrix, i.e euclidean norm by rows (root of sum of squared values)
    
    Arguments:
        input {torch.tensor} -- Nx512 tensor
    
    Keyword Arguments:
        axis {int} -- 1-rows, 0-columns (default: {1})
    
    Returns:
        [type] -- [description]
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

# transforms for mirror image
hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

# transforms for original image
# TODO: get Resize parameters from function
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    """Horizontal Flip
    
    Arguments:
        imgs_tensor {torch.tensor} -- tensor
    
    Returns:
        torch.tensor -- tensor for flipped images
    """
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs


def extract_one_embedding(image, backbone, model_root, size=112,
                    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta=True):
    """Extract features from single image
    
    Arguments:
        image {PIL.Image} -- image for feature extraction
        backbone {backbone object} -- model from backbone module
        model_root {.pth file} -- backbone state file
    
    Keyword Arguments:
        size {int} -- width and height of input image
        device {torch.device} -- which processor to use (default: {torch.device("cuda:0" if torch.cuda.is_available() else "cpu")})
        tta {bool} -- test time augmentations; if True use summarize vector of original and mirror images (default: {True})
    
    Returns:
        np.array -- features
    """
    image = image.convert('RGB')
    image = transform(image)
    ccropped = np.reshape(image, [1, 3, size, size])
    flipped = hflip(image)
    flipped = np.reshape(flipped, [1, 3, size, size])

    # load backbone from a checkpoint
    model = torch.load(model_root)
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
