# Phase 2: independent risk gateway and broker reconciliation

The risk gateway is a separate, unprivileged container on its own Docker-internal network. It
has no broker credentials and no route to the public network. A systemd socket proxy binds only
to `127.0.0.1:18765` and forwards host requests to the fixed gateway address on that internal
bridge. The Docker container has no published host port. The autonomous runner remains
PAPER-only. The reconciler can read Alpaca PAPER or LIVE account feeds, but its client implements
GET requests only.

The gateway rejects malformed, oversized, unprotected, unallowlisted, over-risk, over-notional,
and insufficient-reward/risk order batches. A stale heartbeat, failed reconciliation, audit-log
failure, invalid response, or unreachable endpoint blocks execution. A reconciliation failure
persists a gateway latch; it cannot clear automatically when data later looks healthy.

Audit records are mode-600 JSONL, append-only through the service, hash chained, and checked
against a separate tail anchor on startup. This makes edits, deletion, and truncation detectable;
filesystem administrators can still alter files, so retain host backups and restrict root access.

## 1. Configure separate credentials

On the OCI host, generate two different random tokens. Do not paste their values into chat or
commit them:

```bash
cd /home/ubuntu/apex_quant
umask 077
cp deploy/phase2/risk-gate.env.example deploy/phase2/risk-gate.env
openssl rand -hex 32
openssl rand -hex 32
sudoedit deploy/phase2/risk-gate.env
chmod 600 deploy/phase2/risk-gate.env
```

Set the generated `RISK_GATE_TOKEN` and `RISK_ADMIN_TOKEN` values in both
`deploy/phase2/risk-gate.env` and `/home/ubuntu/apex_quant/.env`. Set the non-secret
`RISK_GATE_URL` in both files; the runner's systemd unit reads the Phase 2 environment file,
while the application settings loader reads `.env`:

```dotenv
RISK_GATE_URL=http://127.0.0.1:18765
RISK_RECONCILIATION_INTERVAL_SECONDS=60
```

Keep the trading configuration at `TRADING_MODE=PAPER` and `BROKER=ALPACA`. The risk service
defaults to the `SPY,QQQ,NVDA` allowlist, 0.5% maximum trade risk, 20% maximum position notional,
three open positions, and a 1.5 aggregate reward/risk floor. The engine rejects settings above
the hard-coded 1% risk, 20% notional, and three-position limits. If the dynamic universe selects
symbols outside the explicit risk allowlist, those orders are rejected until an operator
deliberately updates the allowlist.

## 2. Start the isolated risk container

Use the existing Compose project and Phase 0/1 files so this overlay preserves the deployed
volumes and networks:

```bash
cd /home/ubuntu/apex_quant
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml \
  config -q
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml \
  up -d --build risk-engine
docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml \
  ps risk-engine
curl --fail --silent http://172.20.0.2:8765/health
```

The health response must say `ready`, and Docker must report the container healthy. Verify the
host proxy is loopback-only, while Docker publishes no risk port:

```bash
sudo install -o root -g root -m 644 deploy/phase2/apex-risk-gateway-proxy.service \
  /etc/systemd/system/
sudo install -o root -g root -m 644 deploy/phase2/apex-risk-gateway-proxy.socket \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-risk-gateway-proxy.socket
curl --fail --silent http://127.0.0.1:18765/health
sudo ss -ltnp | grep '127.0.0.1:18765'
docker port "$(docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml ps -q risk-engine)" 8765
sudo ss -ltnp | grep 18765
```

Confirm the network is internal and the container has no published host ports:

```bash
docker network inspect apex-risk-internal \
  --format '{{.Internal}} {{range .IPAM.Config}}{{.Subnet}}{{end}}'
docker port "$(docker compose -p apex-project --env-file deploy/phase0/deployment.env \
  -f deploy/phase0/docker-compose.yml \
  -f deploy/phase1/docker-compose.phase1.yml \
  -f deploy/phase2/docker-compose.phase2.yml ps -q risk-engine)"
```

The host listener must be `127.0.0.1:18765`, the network must report
`true 172.20.0.0/16`, and `docker port` must print no mappings. Do not open port 8765 or 18765
in OCI security lists, host firewalls, or Nginx. The container has no broker keys, no
public-network attachment, a read-only root filesystem, dropped Linux capabilities, and only
the required logs bind mount.

## 3. Require the gate before running the trader

The runner probes the risk heartbeat at the start of every cycle, before contacting the broker.
It sends every entry batch and every stop update, cancellation, or close through the gateway.
An authorization rejection skips that entry; gateway loss or an invalid heartbeat latches the
runner halted for the life of that process. Existing Alpaca bracket legs remain at the broker.
Restart the runner only after the risk container is healthy and the incident is understood.

Verify readiness without submitting an order:

```bash
cd /home/ubuntu/apex_quant
/home/ubuntu/miniforge3/envs/apex_env/bin/python -m server.autonomous_runner --dry-run
```

The dry run must report `Independent risk engine heartbeat: OK`. Start/restart the existing
autonomous-runner systemd unit only after this check passes. The host currently has no runner
unit installed, so install the Phase 2 unit only after the risk tokens and gateway are configured:

```bash
sudo install -o root -g root -m 644 \
  deploy/phase2/apex-autonomous-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-autonomous-runner.service
sudo systemctl --no-pager status apex-autonomous-runner.service
sudo journalctl -u apex-autonomous-runner.service -n 80 --no-pager
```

The unit performs an authenticated risk heartbeat check before starting. If the runner latched
halted after gateway loss, restore and verify the gateway, then restart this specific unit.
Expected heartbeat file: `logs/runner_heartbeat.json`; the dashboard treats data older than
120 seconds as disconnected.

## 4. Enable automated PAPER/LIVE read-only reconciliation

`server.reconciliation` selects the Alpaca endpoint from `TRADING_MODE`, reads account and
positions using a GET-only client, and sends a credential-free snapshot to the risk gateway.
LIVE mode remains subject to the existing `LIVE_CONFIRM` guard; this timer never submits or
modifies orders. On the active PAPER runner, a matching reconciliation also compares target
positions with runner-managed symbols. Every result is appended to the risk audit log. A balance
mismatch, blocked account, malformed position, or unmanaged target holding latches the gateway.

Install and start its once-per-minute timer:

```bash
sudo install -o root -g root -m 644 \
  deploy/phase2/apex-reconcile.service /etc/systemd/system/
sudo install -o root -g root -m 644 \
  deploy/phase2/apex-reconcile.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-reconcile.timer
sudo systemctl start apex-reconcile.service
sudo systemctl --no-pager status apex-reconcile.service
sudo journalctl -u apex-reconcile.service -n 50 --no-pager
```

## 5. Inspect audit integrity and clear a reviewed latch

Risk checks, rejections, reconciliation results, rejected override attempts, and approved
operator latch resets are written to `logs/risk_audit.jsonl`. Verify its hash chain and tail
anchor without printing the records:

```bash
/home/ubuntu/miniforge3/envs/apex_env/bin/python -c \
  "from server.risk_engine import RiskAuditLog; a=RiskAuditLog('logs/risk_audit.jsonl'); print('audit chain OK:', a.sequence, 'entries')"
```

Never delete or edit the audit log or its `.anchor` file. A mismatch prevents the risk service
from starting. After investigating and correcting a reconciliation incident, clear the persistent
risk latch only with an operator identity and a specific reason:

```bash
/home/ubuntu/miniforge3/envs/apex_env/bin/python -m server.risk_engine reset \
  --actor "operator-name" \
  --reason "Explain the reconciled discrepancy and reviewed evidence"
```

The reset is itself recorded as an override in the hash-chained audit log. Restart the runner
after a latch reset; do not treat a successful reset as permission to enable LIVE order routing.
