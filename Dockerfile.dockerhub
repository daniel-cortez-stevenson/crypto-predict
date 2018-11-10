FROM python:3.5.5-jessie

WORKDIR /app

ADD requirements.txt /app/requirements.txt

# Install packages
RUN pip install -U pip setuptools wheel
RUN pip install -U -r requirements.txt

ADD crypr /app/crypr
ADD setup.py /app/setup.py
RUN pip install -e .

ADD models /app/models

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s \
  CMD curl -f http://0.0.0.0:5000/predict?coin=BTC || exit 1

CMD ["python", "/app/crypr/app.py"]
