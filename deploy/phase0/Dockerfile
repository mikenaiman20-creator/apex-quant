ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    APEX_APP_ROOT=/app \
    APEX_RUNTIME_UID=10001 \
    APEX_RUNTIME_GID=10001

WORKDIR /app
COPY deploy/phase0/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.txt \
    && useradd --system --uid 10001 --create-home app

COPY app.py /app/app.py
COPY deploy/entrypoint.py /usr/local/bin/apex-entrypoint.py
RUN chmod 0755 /usr/local/bin/apex-entrypoint.py

# The entrypoint performs a best-effort chmod/chown of bind-mounted logs/ and
# data/, then drops to APEX_RUNTIME_UID before executing Streamlit.
ENTRYPOINT ["python", "/usr/local/bin/apex-entrypoint.py"]

EXPOSE 8501
