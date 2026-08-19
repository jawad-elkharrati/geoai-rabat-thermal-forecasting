FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[parquet,real-data]"

COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
COPY artifacts ./artifacts
COPY reports ./reports

ENTRYPOINT ["python", "-m", "geoai_rabat.cli"]
CMD ["verify-final", "--config", "configs/rabat_real_pilot.json"]
