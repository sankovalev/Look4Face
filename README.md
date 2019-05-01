# Look4Face
Demo of Face Recognition web service.

---
![One face](https://github.com/sankovalev/Look4Face/blob/master/Look4Face/media/examples/Example1.gif)

## Briefly
- The web wrapper is implemented using [Django 2.2](https://docs.djangoproject.com/en/2.2/releases/2.2/).
- [LFW](http://vis-www.cs.umass.edu/lfw/) is used as a main dataset.
- [PyTorch](https://pytorch.org/) for working with neural networks.
- [Faiss](https://github.com/facebookresearch/faiss) for ANN search + vector quantization.
- ResNet50 pretrained on [MS1M-Arcface](https://github.com/deepinsight/insightface#train).
- For **Linux** only.

![Multiple faces](https://github.com/sankovalev/Look4Face/blob/master/Look4Face/media/examples/Example2.gif)

## Installation
Look4Face requires **CUDA** for efficient GPU computations.
This project has a virtual environment that already contains all the necessary packages.

Clone this repository and activate virtual env from root folder:
```sh
$ source bin/activate
```
Start web server
```sh
$ cd Look4Face
$ python manage.py runserver
```
Open [127.0.0.1:8000](http://127.0.0.1:8000) with browser.

### Installation issues
If you receive an error related to Faiss, please visit [this link](https://github.com/onfido/faiss_prebuilt) and install additional packages.

### Useful links 
| Repo | Link |
| ------ | ------ |
| face.evoLVe.PyTorch | github.com/ZhaoJ9014/face.evoLVe.PyTorch |
| facenet_pytorch | github.com/liorshk/facenet_pytorch |
| arcface-pytorch | github.com/ronghuaiyang/arcface-pytorch |
| insightface | github.com/deepinsight/insightface |

### How to use with other data
I strongly recommend to use the [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) repository for train your own models.
1. Put your dataset to Look4Face/datasets with structure:
```
  Look4Face/datasets/your_db/
                      -> id1/
                          -> 1.jpg
                          -> ...
                      -> id2/
                          -> 1.jpg
                          -> ...
                      -> ...
                          -> ...
                          -> ...
  ```
and set variable in Look4Face/Look4Face/settings.py:
```python
DATASET_FOLDER = 'your_db'
```
2. Similarly, put your Faiss index and meta information for labels (dict with pairs id:PersonName) to the same folder, set variables:
```python
DATASET_INDEX = 'your_index.bin'
DATASET_LABELS = 'your_labels.pkl'
```
3. Update Backbone.pth if you need:
```
  Look4Face/backbone/
                     -> your_backbone.pth
  ```
and set variable in Look4Face/Look4Face/settings.py:
```python
BACKBONE_FILE = 'your_backbone.pth'
```

### Todos

 - Write notification if there are no faces on photo.
 - Write tutorial about using own dataset.

### License
MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [L4F]: <https://github.com/sankovalev/Look4Facer>
