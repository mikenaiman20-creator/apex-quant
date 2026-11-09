# Phase 0 deployment: isolated ChromaDB and authenticated dashboard

> The self-signed IP/TLS instructions below are superseded for the current host by the Phase 1
> Let's Encrypt and DNS deployment in [`../phase1/README.md`](../phase1/README.md).

This deployment publishes only Nginx on ports 80/443. Streamlit and Ollama have no host port
mapping. Ollama runs on a separate internal-only network shared only with Streamlit, and is
limited to one CPU and 3 GiB of memory. Its model volume persists across container restarts.
ChromaDB is an opt-in Compose profile attached to a separate, internal-only Docker network with
no other containers attached and no host port mapping. Do not connect application containers to
ChromaDB until its version/runtime has a verified mitigation for the cited issue. Nginx and
OAuth2 Proxy join `apex-internal`; OAuth2 Proxy additionally needs outbound access to GitHub on
`apex-egress` to complete OAuth. Neither container publishes a host port.
The current OCI Docker Snap runtime returns `operation not permitted` when containers use
`no-new-privileges:true`, so this deployment omits that option. Validate runtime support before
reintroducing it; Streamlit still uses a read-only filesystem, a non-root UID, and dropped
capabilities.

The CVE-2026-45829 record currently describes affected ChromaDB versions as 1.0.0 and later. Its
referenced report says the Python backend can execute an untrusted embedding model before the
authorization check; network isolation reduces reachability but is not a patch. Review the
[CVE record](https://cveawg.mitre.org/api/cve/CVE-2026-45829) and
[Chroma issue](https://github.com/chroma-core/chroma/issues/6717) before enabling the
`isolated-chroma` profile. The profile remains opt-in so the default deployment does not start
Chroma inadvertently.

The OAuth allowlist is the exact address in `allowed_emails`; do not replace it with `*`. The
dashboard reads only mounted `data/` and `logs/` files, both read-only, and has no Alpaca API
credentials. This is a read-only operational view, not a Streamlit port of all custom dashboard
functionality.

## Prepare on the OCI host

Copy `app.py` and `deploy/phase0/` to `/home/ubuntu/apex_quant/`. Keep the existing production
`data/` and `logs/` directories on the VPS; do not overwrite them with workstation copies. The
dashboard source, data, and logs are mounted read-only. Keep `.env`, SSH keys, OAuth
secrets, and TLS private keys out of source control. The Compose file requires image references
pinned to verified SHA-256 digests; do not use `:latest`. Select current patched images after
checking vendor advisories and image metadata, then create the deployment environment file from
the template and replace every example digest:

```bash
cp deploy/phase0/deployment.env.example deploy/phase0/deployment.env
chmod 600 deploy/phase0/deployment.env
```

The file must contain these settings:

```dotenv
CHROMA_IMAGE=chromadb/chroma@sha256:<verified-digest>
NGINX_IMAGE=nginx@sha256:<verified-digest>
OAUTH2_PROXY_IMAGE=quay.io/oauth2-proxy/oauth2-proxy@sha256:<verified-digest>
PYTHON_IMAGE=python:3.11-slim@sha256:<verified-digest>
APEX_UI_OLLAMA_URL=http://ollama:11434
APEX_UI_OLLAMA_MODEL=qwen2.5-coder:1.5b
OLLAMA_IMAGE=ollama/ollama@sha256:55d6d58fce77338d8b626abbbbec63b79deeb24db3fd67b8a99951a84083552f
```

The example digests are intentionally invalid. Replace each value with an actual registry digest.
Create
`/home/ubuntu/apex_quant/deploy/phase0/deployment.env`:
Create the OAuth environment file on the VPS and restrict its permissions:

```bash
cd /home/ubuntu/apex_quant
cp deploy/phase0/oauth2-proxy.env.example deploy/phase0/oauth2-proxy.env
chmod 600 deploy/phase0/oauth2-proxy.env
```

Register a GitHub OAuth App with callback URL
`https://151.145.93.209/oauth2/callback`, then set the client ID and secret in
`deploy/phase0/oauth2-proxy.env`. Generate the cookie secret on the host with
`openssl rand -base64 32 | tr -- '+/' '-_' | tr -d '\n'` and set
`OAUTH2_PROXY_COOKIE_SECRET` to its output. Never commit or send these values.

The supplied IP address does not have a publicly trusted certificate in this setup. Generate a
local CA and IP-SAN server certificate on the VPS:

```bash
cd /home/ubuntu/apex_quant
bash deploy/phase0/generate_local_tls.sh
```

Install `deploy/phase0/certs/ca.crt` into the trusted root store of each authorized client using a
trusted transfer channel. Never copy `ca.key` off the VPS. This private-CA approach is for
controlled access; a DNS name and publicly trusted certificate are preferable for production.

## Deploy and verify

Allow TCP/80 and TCP/443 in OCI ingress as needed. Keep TCP/22 restricted to the approved
`37.142.154.0/32` source (or update it if the trusted egress address changes).

```bash
cd /home/ubuntu/apex_quant
docker network inspect apex-internal >/dev/null 2>&1 || \
  docker network create --internal apex-internal
test "$(docker network inspect apex-internal --format '{{.Internal}}')" = true
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml config
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml up --build -d
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml ps
docker network inspect apex-internal
```

`app.py` is bind-mounted read-only. Recreate Streamlit after replacing the host file so the
container does not retain a stale bind-mounted inode. Changing Python dependencies still
requires rebuilding the image. Download the local RAG model and then remove Ollama's temporary
egress attachment:

```bash
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml up -d ollama streamlit
docker network connect apex-egress apex-project-ollama-1
docker exec apex-project-ollama-1 ollama pull qwen2.5-coder:1.5b
docker network disconnect apex-egress apex-project-ollama-1
docker exec apex-project-ollama-1 ollama list
docker inspect apex-project-ollama-1 --format '{{json .NetworkSettings.Ports}}'
docker network inspect apex-llm-internal
```

The temporary egress attachment is only for model download. Keep it disconnected afterward.
The RAG UI uses only the local Ollama model and requires no paid API credentials. Setting
`CRYPTO_SYMBOLS=BTC/USD,ETH/USD` enables public market-data research for the two dashboard
widgets; crypto order routing remains disabled.

The root
`.dockerignore` restricts the image build context to the dashboard and its dependency manifest.

`docker network inspect apex-internal` must report `"Internal": true`. To start Chroma only after
reviewing its runtime/version, add `--profile isolated-chroma` to the `up` command. Then
`docker network inspect apex-chroma-isolated` must also report `"Internal": true`, and no
container other than Chroma may be attached. Chroma's published port map must be empty/null.
Only after enabling that profile, verify it with:

```bash
docker ps --filter name=chromadb
docker inspect chromadb --format '{{json .NetworkSettings.Ports}}'
docker network inspect apex-chroma-isolated
```

Validate the edge behavior (before trusting the CA, `-k` skips certificate
verification; after installing it, omit `-k`):

```bash
curl -k -sS -o /dev/null -w '%{http_code}\n' https://151.145.93.209/
curl -k -sSI https://151.145.93.209/ | head
```

An unauthenticated browser request is expected to redirect to GitHub's OAuth flow (HTTP 302),
not return a bare 401; OAuth2 Proxy uses that redirect to begin authentication. After login with
the allowlisted GitHub email, the dashboard should load. An authenticated but non-allowlisted
account must be denied.

The paper account panel is populated from `logs/paper_dashboard_snapshot.json`, written by the
runner's read-only API polling path. Restart the runner with the updated code and confirm this
file's timestamp advances before expecting equity, positions, and orders in the dashboard.

Verify that Streamlit and ChromaDB are not reachable from outside the VM:

```bash
sudo ss -lntp | grep -E ':(8501|8000)\b' && echo "Unexpected published service port" && exit 1 || true
docker port chromadb
docker port "$(docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml ps -q streamlit)"
```

Do not expose Docker's TCP API or add `ports:` to Streamlit/ChromaDB. A host-level listener check
and OCI security-list/NSG review are both required; Docker network isolation alone does not replace
host firewall and cloud ingress controls.
