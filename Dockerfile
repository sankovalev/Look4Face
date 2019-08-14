FROM ubuntu:18.04
RUN apt-get update && apt-get -y install curl git git-lfs python3 python3-pip \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
    && cd home && git init . \
    && git clone https://github.com/sankovalev/Look4Face.git \
    && git lfs install \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get install -y libopenblas-base \
    && pip3 install virtualenv \
    && virtualenv -p python3 Look4Face && cd Look4Face && git lfs pull \
    && source bin/activate \
    && pip3 install -r requirements.txt && cd Look4Face
CMD ["0.0.0.0:8000"]
ENTRYPOINT ["python3", "manage.py", "runserver"]

#======
#sudo docker build -t l4fimage .
#sudo docker run -d -w home/Look4Face/Look4Face --name look4face -p 8000:8000 l4fimage 
