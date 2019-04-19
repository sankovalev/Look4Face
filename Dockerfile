FROM python:3.7-slim
COPY Look4Face /opt/Look4Face
COPY requirements.txt /opt/requirements.txt
RUN apt-get update && apt-get -y install curl git unzip \
    && apt-get install -y libsm6 libxext6 libxrender-dev libomp-dev \
    && apt-get install -y libopenblas-base libglib2.0-0 \
    && pip install -r /opt/requirements.txt
RUN cd /opt/Look4Face/backbone/ && sh load_backbone.sh 
RUN cd /opt/Look4Face/media/media_root/dataset/ && sh load_dataset.sh
CMD ["0.0.0.0:8000"]
ENTRYPOINT ["python", "/opt/Look4Face/manage.py", "runserver"]

#======
#sudo docker build -t l4fimage .
#sudo docker run -d --name look4face -p 8000:8000 l4fimage 
