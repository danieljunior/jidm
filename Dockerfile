FROM python:3.7.15-slim-bullseye
LABEL author=DanielJunior email="danieljunior@id.uff.br"

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y install gcc g++ nano build-essential libprocps-dev procps locales

ENV LANG pt-BR.UTF-8
ENV LANGUAGE pt-BR.UTF-8

RUN mkdir -p /app

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt 

RUN rm -rf /var/lib/apt/lists/* && apt-get clean
