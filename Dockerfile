FROM ubuntu:18.04
RUN apt-get update && apt-get -y install curl git git-lfs python3 python3-pip \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
    && cd /opt \
    && git clone https://github.com/sankovalev/Look4Face.git \
    && cd /opt/Look4Face \
    && git lfs install \
    && apt-get install -y libsm6 libxext6 libxrender-dev libomp-dev \
    && apt-get install -y libopenblas-base \
    && git lfs pull \
    && pip3 install -r /opt/Look4Face/requirements.txt
CMD ["0.0.0.0:8000"]
ENTRYPOINT ["python3", "/opt/Look4Face/Look4Face/manage.py", "runserver"]

#======
#sudo docker build -t l4fimage .
#sudo docker run -d --name look4face -p 8000:8000 l4fimage 
