
FROM python:3.14.0-slim

RUN apt-get update

WORKDIR /code
COPY . /code

RUN pip install -U pip && pip install -r requirements.txt