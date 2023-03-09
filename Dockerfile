FROM continuumio/anaconda3:2022.10
LABEL author=DanielJunior email="danieljunior@id.uff.br"

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y install gcc g++ nano build-essential libprocps-dev procps locales curl

ENV LANG pt-BR.UTF-8
ENV LANGUAGE pt-BR.UTF-8

RUN mkdir -p /app

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

RUN conda create -y -n jidm && \
    conda init bash && \
    conda activate jidm && \
    conda install pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt

RUN rm -rf /var/lib/apt/lists/* && apt-get clean

