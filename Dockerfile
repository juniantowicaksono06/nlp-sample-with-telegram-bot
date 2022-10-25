FROM python:3.8.15-slim-buster

RUN mkdir -p /src/app

WORKDIR /src/app

# RUN apk update
# RUN apk add nodejs
# RUN apk add npm
# RUN npm install nodemon

# RUN pip install python-telegram-bot tensorflow nlp nltk

RUN apt update
RUN apt install -y nodejs npm
RUN npm install nodemon

RUN pip install python-telegram-bot tensorflow nlp nltk

RUN mkdir -p /setup_ntlk/

COPY ./scripts/main.py /setup_ntlk/main.py

ENV http_proxy="http://22190825:re0_emili4@10.59.82.1:8080"
ENV https_proxy="http://22190825:re0_emili4@10.59.82.1:8080"

RUN npm install -g nodemon

RUN pip install python-dotenv

RUN pip install numpy

RUN python /setup_ntlk/main.py

RUN pip install PySastrawi