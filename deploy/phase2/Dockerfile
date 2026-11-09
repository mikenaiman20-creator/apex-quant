ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN useradd --system --uid 10001 --create-home risk

COPY server/__init__.py /app/server/__init__.py
COPY server/reconciliation.py /app/server/reconciliation.py
COPY server/risk_engine.py /app/server/risk_engine.py

USER 10001:10001
EXPOSE 8765
CMD ["python", "-m", "server.risk_engine", "serve", "--host", "0.0.0.0", "--port", "8765"]
