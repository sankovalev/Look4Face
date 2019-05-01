from django.shortcuts import render, redirect
from django.conf import settings
from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import faiss
from collections import Counter
from util.extract_features import extract_one_embedding
from backbone.model_resnet import ResNet_50
import numpy as np
import logging
import operator
import os
import pickle
import random
import datetime
logging.basicConfig(filename="look4face.log", level=logging.INFO)
MEDIA_PATH = settings.MEDIA_ROOT
DATASET_PATH = settings.DATASET_DIR
DATASET_NAME = settings.DATASET_FOLDER
DATASET_INDEX = settings.DATASET_INDEX
DATASET_LABELS = settings.DATASET_LABELS
BACKBONE_FILE = settings.BACKBONE_FILE
CROPS_PATH = 'crops'
reference = get_reference_facial_points(default_square = True)


def main(request):
    """Displays the main page
    
    Arguments:
        request {[type]} -- [description]
    """
    # SHOW MAIN PAGE
    if request.method == 'GET':
        context = {
            'title': 'Main',
            }
        return render(request, 'index.html', context)
    # GOT NEW PHOTO
    elif request.method == 'POST':
        # INITIAL SEARCH
        if request.POST.get('refine') == "False":
            image = request.FILES.get('photo')
            image_type = image.name.split('.')[-1] #png/jpg/jpeg
            now = datetime.datetime.now()
            image_path = f'{now.day}{now.month}{now.year}_{now.hour}:{now.minute}:{now.second}.{image_type}'
            full_path = os.path.join(MEDIA_PATH, image_path)
            with open(full_path, 'wb+') as destination:
                destination.write(image.read())
            img = Image.open(full_path)
            _, landmarks = detect_faces(img) #TODO: change onet/rnet/pnet path
            if landmarks == []:
                pass
                # there are no faces on the photo
                # TODO: send message
                return redirect('Main Page')
            count = landmarks.shape[0]
            if count == 1:
                img = align_face(img, landmarks[0]) # cropped aligned face, ready for search
                D, I = search(img) # distances and indexes
                results_dict = results(D,I)

                context = {
                    'title': 'Search Results',
                    'results_dict': results_dict
                }
                return render(request, 'results.html', context)

            else:
                face_urls = refine_face(img, landmarks, image_path)
                context = {
                    'title': 'Choose face',
                    'faces_list': face_urls,
                }
                return render(request, 'refine.html', context)
        # SEARCH AFTER REFINING
        elif request.POST.get('refine') == "True":
            image_path = request.POST.get('imagecrop') # number selected face
            full_path = os.path.join(MEDIA_PATH, image_path)
            img = Image.open(full_path) # cropped aligned face, ready for search
            D, I = search(img) # distances and indexes
            results_dict = results(D,I)

            context = {
                'title': 'Search Results',
                'results_dict': results_dict
            }
            return render(request, 'results.html', context)


def search(img, k=10, nprobe=10):
    backbone = ResNet_50([112,112])
    pth = os.path.join(settings.BACKBONE_DIR, BACKBONE_FILE) # Pretrained backbone for ResNet50
    
    index = faiss.read_index(os.path.join(DATASET_PATH, DATASET_INDEX)) # load index
    index_ivf = faiss.downcast_index(index.index)
    index_ivf.nprobe = nprobe # change nprobe

    query = np.array(extract_one_embedding(img, backbone, pth)).astype('float32').reshape(1,-1)
    D,I = index.search(query, k)
    return D[0], I[0]


def results(D, I):
    """Analyze neighbors
    
    Arguments:
        D {np.array} -- Distances to neighbors
        I {np.array} -- Indexes of neighbors 
    """
    lst = list(I)
    # calculate probabilities
    proba_dict = dict.fromkeys(list(set(lst)), 0.0)
    for i, label in enumerate(lst):
        proba_dict[label] += 1/((i+1)*D[i]) # weight for each neighbour
    proba_dict = {k: v for k, v in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)}
    total = sum(proba_dict.values(), 0.0)
    # read real names and rename keys
    with open(os.path.join(DATASET_PATH, DATASET_LABELS), 'rb') as f:
        names = pickle.load(f) # load real names
    proba_dict = {names[k].replace('_', ' '): [round(v/total*100,2), os.path.join('dataset', DATASET_NAME, str(k), random.choice(os.listdir(os.path.join(DATASET_PATH, 'lfw', str(k)))))] for k, v in proba_dict.items()} # 'name1':[probability1,photo1] ...
    
    return proba_dict


def align_face(img, landmarks, crop_size=112):
    facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    return img_warped


def refine_face(img, landmarks, image_path):
    count = landmarks.shape[0]
    face_urls = []
    for i in range(count):
        face = align_face(img, landmarks[i], crop_size=112)
        face.save(os.path.join(MEDIA_PATH, CROPS_PATH, f'{i}_{image_path}'))
        face_urls.append(os.path.join(CROPS_PATH, f'{i}_{image_path}'))
    return face_urls