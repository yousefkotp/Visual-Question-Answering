FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5000

CMD ["python", "-m", "flask", "run"]

# docker build -t vqa-flask-app .
# docker run -p 5000:5000 vqa-flask-app