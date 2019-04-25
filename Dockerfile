FROM tensorflow/tensorflow:latest-py3

RUN pip install tf-nightly-gpu-2.0-preview==2.0.0.dev20190413
RUN pip install flask
