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
DATASET_PATH = settings.DATASET_DIR
CROPS_PATH = 'crops'
reference = get_reference_facial_points(default_square = True)


def main(request):
    """Displays the main page
    
    Arguments:
        request {[type]} -- [description]
    """
    # logger = logging.getLogger('main')
    # ОТОБРАЖАЕМ СТРАНИЦУ
    if request.method == 'GET':
        # try:
        context = {
            'title': 'Look4Face',
            }
        return render(request, 'index.html', context)
        # except Exception as e:
        #     logger.error(f'GET-request, {str(e)}')
        #     return redirect('Main Page')

    # ЗАГРУЗИЛИ НОВУЮ ФОТКУ
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
            count = landmarks.shape[0]
            if count == 0:
                pass
                return
                # there are no faces on the photo
                # TODO: send message
            elif count == 1:
                img = align_face(img, landmarks[0]) # cropped aligned face, ready for search
                D, I = search(img) # distances and indexes
                print(D,I)

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
            print(D,I)
            return render(request, 'results.html', context)


def search(img, k=10, nprobe=10):
    import mkl
    mkl.get_max_threads()
    import faiss
    from util.extract_features import extract_one_embedding
    from backbone.model_resnet import ResNet_50

    backbone = ResNet_50([112,112])
    pth = os.path.join(settings.BACKBONE_DIR, 'Backbone.pth') # Pretrained backbone for ResNet50
    
    index = faiss.read_index(os.path.join(DATASET_PATH, 'index.bin')) # load index
    index_ivf = faiss.downcast_index(index.index)
    index_ivf.nprobe = nprobe # change nprobe

    query = np.array(extract_one_embedding(img, backbone, pth)).astype('float32').reshape(1,-1)
    D,I = index.search(query, k)
    return D[0], I[0]


def align_face(img, landmarks, crop_size=112):
    facial5points = [[landmarks[j], landmarks[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    return img_warped


def refine_face(img, landmarks, image_path):
    count = landmarks.shape[0]
    face_urls = []
    for i in range(count):
        # face = img.resize((224, 224), box=bounding_boxes[i][:4])
        face = align_face(img, landmarks[i], crop_size=112) # try 224x224
        # face = img.crop(bounding_boxes[i][:4])
        # face = face.resize((175,175))
        face.save(os.path.join(MEDIA_PATH, CROPS_PATH, f'{i}_{image_path}'))
        face_urls.append(os.path.join(CROPS_PATH, f'{i}_{image_path}'))
    return face_urls