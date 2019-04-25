FROM python:3.6.7

RUN pip install tf-nightly-2.0-preview==2.0.0.dev20190413
RUN pip install flask
RUN pip install flask-cors
RUN pip install tensorflow_datasets
RUN pip install matplotlib