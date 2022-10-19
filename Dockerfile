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

RUN python /setup_ntlk/main.py

RUN npm install -g nodemon

RUN pip install python-dotenv