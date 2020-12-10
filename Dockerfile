FROM continuumio/miniconda3

COPY ./src /api/src
COPY environment.yml /environment.yml

RUN apt-get update \
    &&  conda env create -f /environment.yml

SHELL ["conda", "run", "-n", "dockerFastAPI", "/bin/bash/", "-c"]

WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["conda", "run", "-n", "dockerFastAPI", "python", "-m", "src.main"]




