FROM runpod/pytorch:3.10-2.0.1-120-devel

ADD requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN  python3 download.py

ADD . .
