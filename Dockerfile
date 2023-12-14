FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime


ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ADD . .

RUN python3 download.py
