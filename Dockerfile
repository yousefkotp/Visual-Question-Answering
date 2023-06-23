FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git

RUN pip install git+https://github.com/openai/CLIP.git 

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["flask", "run"]