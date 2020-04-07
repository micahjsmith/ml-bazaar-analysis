FROM python:3.6.8

USER root
WORKDIR /work

COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY ./analysis.py .

COPY ./mongodb_config.json .

CMD ["python", "./analysis.py"]
