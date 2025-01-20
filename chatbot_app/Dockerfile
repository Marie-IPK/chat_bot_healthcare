FROM python:3.12.2-slim

RUN apt-get update && apt-get install -y curl && apt-get clean

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader punkt_tab

WORKDIR /app
COPY . /app

RUN python3 model_chat2.py

EXPOSE 8080
CMD ["python3", "chatbot_flask.py"]
