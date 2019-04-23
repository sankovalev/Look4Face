from django.shortcuts import render, redirect
from django.conf import settings
from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import logging
import os
import datetime
logging.basicConfig(filename="look4face.log", level=logging.INFO)
MEDIA_PATH = settings.MEDIA_ROOT
CROPS_PATH = 'crops'

def main(request):
    """Displays the main page
    
    Arguments:
        request {[type]} -- [description]
    """
    logger = logging.getLogger('main')
    # ОТОБРАЖАЕМ СТРАНИЦУ
    if request.method == 'GET':
        # try:
        context = {
            'title': 'Главная страница',
            }
        return render(request, 'index.html', context)
        # except Exception as e:
        #     logger.error(f'GET-request, {str(e)}')
        #     return redirect('Main Page')

    # ЗАГРУЗИЛИ НОВУЮ ФОТКУ
    elif request.method == 'POST':
        image = request.FILES.get('photo')
        image_type = image.name.split('.')[-1] #png/jpg/jpeg
        now = datetime.datetime.now()
        image_path = f'{now.day}{now.month}{now.year}_{now.hour}:{now.minute}:{now.second}.{image_type}'
        full_path = os.path.join(MEDIA_PATH, image_path)
        with open(full_path, 'wb+') as destination:
            destination.write(image.read())
        img = Image.open(full_path)
        bounding_boxes, landmarks = detect_faces(img)
        count = bounding_boxes.shape[0]
        if count == 0:
            pass
            # there are no faces on this photo
            # TODO: send message
        elif count == 1:
            search(img.crop(bounding_boxes[0][:4]), landmarks)
        else:
            refine_face(request, img, bounding_boxes, landmarks, image_path)

        context = {
            'title': str(type(img))
        }        
        return render(request, 'index.html', context)

def search(img, landmarks):
    """Main search function
    
    Arguments:
        img {[type]} -- [description]
        landmarks {[type]} -- [description]
    """


def extract_features(img, landmarks):
    """Extract face features
    
    Arguments:
        img {[type]} -- crop of image with face
    """
    aligned_image = align_face(img, landmarks)


def align_face(img, landmarks):
    pass
    return


def refine_face(request, img, bounding_boxes, landmarks, image_path):
    """Ask which face to look4
    
    Arguments:
        img {[type]} -- [description]
        bounding_boxes {[type]} -- [description]
        landmarks {[type]} -- [description]
    """
    if request.method == 'GET':
        count = bounding_boxes.shape[0]
        face_urls = []
        for i in range(count):
            face = img.crop(bounding_boxes[i][:4])
            face.save(os.path.join(MEDIA_PATH, CROPS_PATH, f'{i}_{image_path}'))
            face_urls.append(os.path.join(CROPS_PATH, f'{i}_{image_path}'))
        context = {
            'title': 'Choose face',
            'faces': face_urls
        }
        return render(request, 'refine.html', context)
    elif request.method == 'POST':
        face_name = request.POST.get('name')