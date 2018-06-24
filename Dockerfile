FROM python:3.5.5-jessie

WORKDIR /app

ADD src /app/src
ADD setup.py /app/setup.py
ADD requirements.txt /app/requirements.txt

RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt

ADD models /app/models

EXPOSE 5000

CMD ["python", "/app/src/app.py"]
