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
        now = datetime.datetime.now()
        image_path = os.path.join(MEDIA_PATH, f'{now.day}{now.month}{now.year}_{now.hour}:{now.minute}:{now.second}_{image.name}')
        with open(image_path, 'wb+') as destination:
            destination.write(image.read())
        
        img = Image.open(image_path)

        context = {
            'title': str(type(img))
        }        
        return render(request, 'index.html', context)


# def detect_faces(image):
#     pass
#     return