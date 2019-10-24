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
- For **Linux** and **Mac OS**.

![Multiple faces](https://github.com/sankovalev/Look4Face/blob/master/Look4Face/media/examples/Example2.gif)

---
## Easy using with Docker
1. Build image from Dockerfile
```sh
$ docker build -t l4fimage .
```

2. Run container as daemon and expose 8000 port
```sh
$ docker run -d --name look4face -p 8000:8000 l4fimage
```

## Installation
If you have CUDA installed, all calculations will be performed on the GPU, otherwise - on the CPU.

1. Install [Git-LFS](https://git-lfs.github.com/)

2. Clone this repository and load LFS objects:
```sh
$ git init .
$ git clone https://github.com/sankovalev/Look4Face.git
$ git lfs install
$ git lfs pull
```

3. Create virtualenv and activate it:
```sh
$ virtualenv -p python3 Look4Face
$ cd Look4Face
$ source bin/activate
```

4. Make sure that you are using python3 & pip3 from virtual environment:
```sh
$ which python3
$ which pip3
```

5. Install all requirements:
```sh
$ pip3 install -r requirements.txt
```

6. Start web server
```sh
$ cd Look4Face
$ python3 manage.py runserver
```
7. Open [127.0.0.1:8000](http://127.0.0.1:8000) with browser.

### Installation issues
If you receive an error related to Faiss, please visit [this link](https://github.com/onfido/faiss_prebuilt) and install additional packages.

### Useful links 
| Repo | Link |
| ------ | ------ |
| face.evoLVe.PyTorch | https://github.com/ZhaoJ9014/face.evoLVe.PyTorch |
| facenet_pytorch | https://github.com/liorshk/facenet_pytorch |
| arcface-pytorch | https://github.com/ronghuaiyang/arcface-pytorch |
| insightface | https://github.com/deepinsight/insightface |
sudo docker build -t l4fimage .
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
