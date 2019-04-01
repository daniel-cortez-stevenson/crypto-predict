ARG PYTHON_VERSION=3.6

FROM python:${PYTHON_VERSION}-slim-stretch

RUN apt-get update && apt-get install -y curl

WORKDIR /app

ADD crypr crypr
ADD setup.py setup.py

RUN pip install -U pip setuptools wheel \
        && pip install -U . \
        && rm -rf /root/.cache/pip/*

ADD models models
ADD docker/.env_docker .env
ADD api api

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=5s \
  CMD curl -f http://0.0.0.0:5000/greeting || exit 1

ENTRYPOINT ["python"]

CMD ["-m", "connexion", "run", "api/swagger.yaml", "-v"]
