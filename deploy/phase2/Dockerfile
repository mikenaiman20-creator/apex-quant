ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APEX_APP_ROOT=/app \
    APEX_RUNTIME_UID=10001 \
    APEX_RUNTIME_GID=10001

WORKDIR /app
RUN useradd --system --uid 10001 --create-home risk

COPY server/__init__.py /app/server/__init__.py
COPY server/reconciliation.py /app/server/reconciliation.py
COPY server/risk_engine.py /app/server/risk_engine.py
COPY server/io_utils.py /app/server/io_utils.py
COPY deploy/entrypoint.py /usr/local/bin/apex-entrypoint.py
RUN chmod 0755 /usr/local/bin/apex-entrypoint.py

# The entrypoint chmod/chowns the shared logs mount and drops to the configured
# runtime UID before starting the gateway.
ENTRYPOINT ["python", "/usr/local/bin/apex-entrypoint.py"]

EXPOSE 8765
CMD ["python", "-m", "server.risk_engine", "serve", "--host", "0.0.0.0", "--port", "8765"]
