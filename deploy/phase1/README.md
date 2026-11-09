# Phase 1 deployment: DNS, trusted TLS, persistence, and full dashboard telemetry

This deployment uses `151.145.93.209.nip.io` for the supplied OCI IPv4 address. `nip.io` is a
third-party wildcard DNS service, not a domain controlled by Apex Quant. Its resolution and
continued use of the same public IP are operational dependencies. Use a domain you own for a
long-lived production endpoint.

The dashboard still receives no Alpaca credentials. The PAPER runner publishes a sanitized
account/positions/orders snapshot; the dashboard reads it and the Knowledge Brain SQLite
database read-only. Source modules are mounted read-only. ChromaDB data uses a persistent named
Docker volume, and quantitative `data/` and `logs/` remain bind-mounted from the production
host. Never run `docker compose down --volumes` for this stack.

## 1. Prepare Phase 1 values and network

Copy this directory and the updated `app.py` and `server/paper_snapshot.py` to
`/home/ubuntu/apex_quant`. Keep the running production `data/` and `logs/` directories intact.
Use Compose project name `apex-project` for every command below; it preserves the existing
`apex-project_chroma_data` named volume across the deployment.

Append these values to the existing, permission-restricted
`deploy/phase0/deployment.env`, replacing the Certbot image example with the verified ARM64
manifest digest:

```dotenv
DASHBOARD_DOMAIN=151.145.93.209.nip.io
CERTBOT_EMAIL=mikenaiman20@gmail.com
CERTBOT_IMAGE=certbot/certbot@sha256:<verified-arm64-manifest-digest>
APEX_RUNTIME_UID=1001
APEX_RUNTIME_GID=1001
```

The digest must identify an official Certbot image manifest that supports the OCI ARM64
architecture. Keep `deployment.env` mode `600`. Existing verified Nginx, OAuth2 Proxy, Python,
and Chroma image digests remain in that same file.
Use Docker Compose v2.24.4 or newer; the Phase 1 overlay replaces the Phase 0 certificate mounts
so the temporary ACME setup can start before a certificate exists.
The Streamlit container runs as this unprivileged host UID/GID so it can read the mode-`700`
production data and log directories without making those files world-readable. Verify the values
against `id -u ubuntu` and `id -g ubuntu` before starting the stack.

Resolve the hostname on the OCI host and confirm it returns `151.145.93.209`:

```bash
getent ahostsv4 151.145.93.209.nip.io
```

In the OCI NSG/security list, allow inbound TCP/80 and TCP/443. Do not allow TCP/8501, TCP/8000,
or the Docker API. Keep SSH ingress restricted to the approved administrator source CIDR.
Before starting the Phase 1 Compose stack, check that the existing Nginx/Compose deployment is the
only process binding public ports 80/443; do not terminate an unidentified listener.

## 2. Update GitHub OAuth callback

In the GitHub OAuth App settings, set the Authorization callback URL to exactly:

```text
https://151.145.93.209.nip.io/oauth2/callback
```

Update the existing server-side `deploy/phase0/oauth2-proxy.env` with the same redirect URL,
cookie domain, and whitelist. Retain the existing client ID, secret, and cookie secret; do not copy
them to the dashboard container or into source control. The callback scheme must match the public
dashboard scheme exactly (`https://` here); mixing `http://` and `https://` causes OAuth state
failures and browser certificate warnings. Nginx forwards `X-Forwarded-Proto`, `X-Forwarded-Host`,
and `X-Forwarded-Port`, and oauth2-proxy runs with `--reverse-proxy=true` and a secure cookie.

```sh
OAUTH2_PROXY_REDIRECT_URL=https://151.145.93.209.nip.io/oauth2/callback
OAUTH2_PROXY_COOKIE_DOMAIN=151.145.93.209.nip.io
OAUTH2_PROXY_WHITELIST_DOMAINS=.151.145.93.209.nip.io
OAUTH2_PROXY_COOKIE_SECURE=true
```

`nip.io` is a third-party wildcard DNS service. The name resolves to the embedded IP, but its
certificate and OAuth flow depend on that service and on the same public IP. Use a domain you
control for long-lived production deployments.

## 3. Issue a Let's Encrypt certificate

Certbot HTTP-01 validation needs public TCP/80 to reach Nginx. Prepare persistent certificate
storage and the webroot:

```bash
cd /home/ubuntu/apex_quant
sudo install -d -o root -g root -m 755 /etc/letsencrypt
sudo install -d -o ubuntu -g ubuntu -m 755 /home/ubuntu/apex_quant/.certbot-webroot
```

Start the temporary HTTP-only Nginx config, which serves ACME challenges and returns 503 for
other paths while the certificate is being issued:

```bash
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase1/docker-compose.acme.yml \
  up -d nginx

docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase1/docker-compose.acme.yml \
  --profile certbot run --rm certbot
```

Confirm these files exist before switching to the HTTPS config:

```bash
sudo test -s /etc/letsencrypt/live/151.145.93.209.nip.io/fullchain.pem
sudo test -s /etc/letsencrypt/live/151.145.93.209.nip.io/privkey.pem
```

Start the authenticated HTTPS Nginx config and application:

```bash
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  up -d --build
```

## 4. Enable certificate renewal

Install the provided systemd service and timer after the HTTPS stack is running:

```bash
sudo install -o root -g root -m 644 \
  deploy/phase1/apex-certbot-renew.service /etc/systemd/system/
sudo install -o root -g root -m 644 \
  deploy/phase1/apex-certbot-renew.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-certbot-renew.timer
sudo systemctl list-timers apex-certbot-renew.timer --no-pager
```

The renewal systemd service runs as root to access the Docker socket without granting the
`ubuntu` account broad Docker-group privileges. Verify Docker access with `sudo docker info`
before enabling the timer.

The timer runs `certbot renew` twice daily and reloads Nginx only after a successful renewal
command. Check its first dry run without changing certificates:

```bash
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  --profile certbot run --rm certbot renew --dry-run \
  --webroot --webroot-path /var/www/certbot
```

## 5. Persist ChromaDB and production data

The Compose stack stores ChromaDB under the named `chroma_data` volume and maps the host's
`/home/ubuntu/apex_quant/data` and `logs` directories read-only into Streamlit. The code modules
(`backtest/`, `config/`, `engine/`, `server/`) are also mounted read-only. Verify mounts:

```bash
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  ps
docker inspect "$(docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml ps -q chromadb)" \
  --format '{{json .Mounts}}'
docker volume ls
```

Chroma remains on its own internal-only network and has no published host port. Enable its
`isolated-chroma` profile only after confirming the deployed Chroma runtime is not the vulnerable
Python backend described by CVE-2026-45829, or that the exact image is vendor-fixed. Isolation is
not a software patch. Do not attach other services to its network or add a `ports:` mapping.

## 6. Deploy sanitized PAPER telemetry

The paper snapshot service makes only GET requests to the Alpaca PAPER API. It refuses to run
unless `TRADING_MODE=PAPER` and `BROKER=ALPACA`, publishes an allowlisted snapshot to `logs/`,
and never gives broker credentials to Streamlit. It is not a trading runner and cannot place,
replace, or cancel orders.

Install the service and its once-per-minute timer:

```bash
sudo install -o root -g root -m 644 \
  deploy/phase1/apex-paper-snapshot.service /etc/systemd/system/
sudo install -o root -g root -m 644 \
  deploy/phase1/apex-paper-snapshot.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start apex-paper-snapshot.service
sudo systemctl enable --now apex-paper-snapshot.timer
sudo systemctl --no-pager status apex-paper-snapshot.service
sudo systemctl list-timers apex-paper-snapshot.timer --no-pager
```

Verify the sanitized publication and local database integrity:

```bash
python3 -c "import json,pathlib; p=pathlib.Path('logs/paper_dashboard_snapshot.json'); d=json.loads(p.read_text()); print(d['timestamp'], d['paper_api_status'], d['mode'], len(d['positions']), len(d['orders']))"
python3 -c "import sqlite3; c=sqlite3.connect('file:data/knowledge_brain.sqlite?mode=ro',uri=True); print(c.execute('PRAGMA quick_check').fetchone()[0]); c.close()"
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  exec -T streamlit python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').read().decode())"
```

The Research tab reads the SQLite database read-only and displays strategy/source counts. The
Trading tab reports account and order state only from the sanitized snapshot; stale data is
explicitly flagged.

## 7. Acceptance checks

Run these checks from a machine outside the OCI VM. A no-auth request should redirect to GitHub
OAuth (HTTP 302); after GitHub sign-in with the allowlisted email, the dashboard should load:

```bash
curl -sSI http://151.145.93.209.nip.io/
curl -sSI https://151.145.93.209.nip.io/
```

The HTTP response must redirect to HTTPS. The HTTPS certificate must validate without `-k`.
GitHub callback must complete successfully and a non-allowlisted identity must be denied.
`docker ps` must show no published ports for Streamlit or ChromaDB; OCI ingress and host firewall
must block direct access to 8501/8000. Confirm the paper account snapshot and SQLite integrity
are current in the authenticated dashboard.

The previously observed HTTP IP page is not a passing security check. Do not consider Phase 1 TLS
acceptance complete until the trusted HTTPS URL and OAuth flow pass these checks. This workspace
cannot configure OCI DNS/NSGs, issue certificates on the host, or verify the remote deployment.
