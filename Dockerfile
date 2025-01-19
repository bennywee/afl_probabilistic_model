FROM python:3.11 AS base
RUN apt-get update && apt-get install -y r-base
COPY requirements.txt requirements.r ./
RUN pip install -r requirements.txt 
RUN Rscript requirements.r
RUN rm requirements.txt requirements.r
WORKDIR /afl_probabilistic_model

FROM base AS dev
ENTRYPOINT ["/bin/bash"]