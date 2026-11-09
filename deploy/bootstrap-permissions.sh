#!/bin/sh
# Host-side permission bootstrap for Apex Quant bind-mounted runtime directories.
#
# Run as root (systemd ExecStartPre=+... or sudo). Bind-mounted logs/ and data/
# are shared by the host runner, the risk container, and the read-only Streamlit
# dashboard. They must be readable by the container runtime UID even when that
# UID differs from the host owner.
#
# Sensitive risk-gateway files stay mode 0600; only ownership is normalized.
set -eu

APP_ROOT="${APEX_APP_ROOT:-/home/ubuntu/apex_quant}"
RUNTIME_UID="${APEX_RUNTIME_UID:-1001}"
RUNTIME_GID="${APEX_RUNTIME_GID:-1001}"

mkdir -p "$APP_ROOT/logs" "$APP_ROOT/data" "$APP_ROOT/data/staging"
chown -R "$RUNTIME_UID:$RUNTIME_GID" "$APP_ROOT/logs" "$APP_ROOT/data"

# Directories: owner/group writable and traversable; files: shared readable and
# writable so either the host service or the container can rotate them.
find "$APP_ROOT/logs" "$APP_ROOT/data" -type d -exec chmod 0777 {} +
find "$APP_ROOT/logs" "$APP_ROOT/data" -type f \
    ! -name 'risk_audit.jsonl' \
    ! -name 'risk_audit.jsonl.anchor' \
    ! -name 'risk_engine_state.json' \
    -exec chmod 0666 {} +

# Re-assert the private mode for the tamper-evident risk gateway files.
for sensitive in \
    "$APP_ROOT/logs/risk_audit.jsonl" \
    "$APP_ROOT/logs/risk_audit.jsonl.anchor" \
    "$APP_ROOT/logs/risk_engine_state.json"; do
    if [ -f "$sensitive" ]; then
        chmod 0600 "$sensitive"
    fi
done

umask 0002
