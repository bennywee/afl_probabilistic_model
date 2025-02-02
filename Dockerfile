FROM python:3.11 AS base
RUN apt-get update && apt-get install -y r-base
COPY requirements.txt requirements.r ./
RUN mkdir ~/.R
RUN touch ~/.R/Makevars
RUN echo "CPICFLAGS=-fPIC \n\
CXXPICFLAGS=-fPIC \n\
CXX11PICFLAGS=-fPIC \n\
CXX14PICFLAGS=-fPIC \n\
CXX17PICFLAGS=-fPIC \n\
CXX20PICFLAGS=-fPIC \n\
FPICFLAGS=-fPIC" >> ~/.R/Makevars
RUN pip install -r requirements.txt 
RUN Rscript requirements.r
RUN rm requirements.txt requirements.r
WORKDIR /afl_probabilistic_model

FROM base AS dev
ENTRYPOINT ["/bin/bash"]