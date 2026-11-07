#!/usr/bin/env bash
set -euo pipefail
umask 077

cd "$(dirname "$0")"
mkdir -p certs
chmod 700 certs

if [[ -e certs/ca.key || -e certs/server.key ]]; then
    echo "Refusing to overwrite existing TLS private keys in deploy/phase0/certs." >&2
    exit 1
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

openssl genrsa -out certs/ca.key 3072
openssl req -x509 -new -sha256 -days 3650 \
    -key certs/ca.key \
    -out certs/ca.crt \
    -subj "/CN=Apex Quant Local Root CA"

openssl genrsa -out certs/server.key 3072
openssl req -new -sha256 \
    -key certs/server.key \
    -out "$workdir/server.csr" \
    -subj "/CN=151.145.93.209"

cat > "$workdir/server.ext" <<'EOF'
subjectAltName=IP:151.145.93.209
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EOF

openssl x509 -req -sha256 -days 90 \
    -in "$workdir/server.csr" \
    -CA certs/ca.crt \
    -CAkey certs/ca.key \
    -CAcreateserial \
    -out certs/server.crt \
    -extfile "$workdir/server.ext"

chmod 600 certs/ca.key certs/server.key
chmod 644 certs/ca.crt certs/server.crt
echo "Created an IP-SAN server certificate. Install certs/ca.crt only on trusted client devices."
echo "Keep certs/ca.key private; rotate certs/server.crt before its 90-day expiry."
