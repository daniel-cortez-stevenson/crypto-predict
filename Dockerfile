FROM python:3.5.5-jessie

WORKDIR /app

# Move necessary components over to container
ADD src /app/src
ADD setup.py /app/setup.py
ADD requirements.txt /app/requirements.txt

# Install packages and upgrade Flask to development version
RUN pip install -U pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -U https://github.com/pallets/flask/archive/master.tar.gz

ADD models /app/models

EXPOSE 5000

CMD ["python", "/app/src/app.py"]
