FROM python:3.5-jessie

WORKDIR /app

# Update pip
RUN pip install -U pip setuptools wheel

# Install crypr
ADD crypr /app/crypr
ADD setup.py /app/setup.py
RUN pip install -U .

ADD models /app/models

ADD docker/.env_docker /app/.env

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s \
  CMD curl -f http://0.0.0.0:5000 || exit 1

CMD ["python", "-c", "exec(open('/app/crypr/api/app.py').read())"]
