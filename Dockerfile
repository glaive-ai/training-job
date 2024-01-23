FROM runpod/pytorch:3.10-2.0.1-120-devel

WORKDIR /workspace

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD download.py download.py
RUN python3 download.py
ARG IMAGE_NAME
ENV IMAGE_NAME=$IMAGE_NAME
ADD . .