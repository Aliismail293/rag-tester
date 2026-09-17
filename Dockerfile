FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY demo ./demo

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["/app/demo/run_demo.sh"]
