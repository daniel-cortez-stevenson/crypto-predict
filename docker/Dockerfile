FROM python:3.5.5-jessie

WORKDIR /app

ADD requirements.txt /app/requirements.txt

# Install packages
RUN pip install -U pip setuptools wheel
RUN pip install -U -r requirements.txt

ADD src /app/src
ADD setup.py /app/setup.py
RUN pip install -e .

ADD models /app/models

EXPOSE 5000

CMD ["python", "/app/src/app.py"]
