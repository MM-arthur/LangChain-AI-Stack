FROM continuumio/anaconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

RUN echo "source activate langgraph-env" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

CMD ["python", "main.py"]
