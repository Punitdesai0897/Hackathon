FROM python:3.11-slim

WORKDIR /app

COPY . ./

RUN pip install flask gunicorn

CMD gunicorn --bind :$PORT app:api