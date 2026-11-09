# ApexQuant - backtest quickstart (paper phase)

1. Install Python 3.10+ from python.org (Windows: tick "Add Python to PATH").
2. Unzip this folder. Open a terminal INSIDE the apex_quant folder.
3. pip install -r requirements.txt
4. For downloading data only, copy .env.example to .env and set your Alpaca API key and secret.
   The offline backtest does not need credentials. Never share or commit .env.
   Configuration loads `.env` by absolute path from the project root (not the shell's current
   directory). Use the exact names `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
5. To download data (free IEX feed):
     python -m backtest.cli fetch --symbols SPY QQQ NVDA --start 2024-01-01 --end 2026-09-30 --out data
6. Run the purged combinatorial validation on existing CSV files (no API keys required):
     python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --walk-forward
   `--csv-dir data --symbols SPY QQQ NVDA` are also the defaults, so
     python -m backtest.cli run
   runs CPCV/DSR plus a full-history evaluation and matched random-entry baseline from those
   local files. CPCV defaults to six chronological groups, three test groups per split, and a
   one-session post-test embargo; boundary-crossing trades are purged and portfolio equity is
   reset at each group boundary. Every invocation adds the number of tested parameter variants
   to `logs/strategy_trial_ledger.json`, and DSR uses that cumulative trial count.
   Add `--walk-forward-only` to skip the full-history and random-entry replays and run only the
   parameter-selection folds plus out-of-sample report. Every run prints PBO, DSR probability,
   current and cumulative trial counts, and whether the CPCV/DSR gates passed. The output also
   includes expectancy in R, win rate, profit factor, maximum drawdown, and CAGR. Change
   `--cpcv-groups` or `--cpcv-embargo-days` only for a documented research design; all CPCV
   settings are included in the report.
   For this dataset, the shorter OOS-only invocation is:
     python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --walk-forward-only --workers 8
   The independent opening-range breakout sleeve uses a prior-20-bar 1.5x relative-volume
   gate, a 14-minute ATR regime above its same-time 10-session average, and a trailing stop
   activated at +1R and spaced at half the opening-range width. Its intentionally coarse
   two-value reward/risk grid is evaluated through the same cost-aware CPCV/DSR and
   walk-forward path:
     python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --strategy orb --walk-forward-only
   The independent short-term mean-reversion sleeve can be tested using its rolling 20-bar
   close z-score (entry at |z| >= 2.0), synchronized SPY panic gate, next-bar fills and
   mandatory same-day liquidation:
     python -m backtest.cli run --csv-dir data --symbols SPY QQQ NVDA --strategy zscore --walk-forward-only
   (Windows: use `py` instead of `python` if `python` is not found.)
Sanity check with no keys needed:
     python -m backtest.cli run --synthetic --symbols SPY --days 160

## Autonomous paper runner

The runner is intentionally paper-only and sends orders only to Alpaca's paper endpoint. Set
`TRADING_MODE=PAPER`, `BROKER=ALPACA`, `ALPACA_API_KEY`, and `ALPACA_API_SECRET` in `.env`.
It schedules its daily scan from Alpaca's next market open (15 minutes before), so it follows US
exchange hours and daylight-saving transitions rather than assuming a fixed Israel-time offset.

The default candidate pool is Alpaca's most-active US equity screen. Optionally add an index
constituent list to `data/universe.csv` (one symbol per line or a `symbol`/`ticker` CSV header);
those symbols are combined with the screen. Each scan filters prior daily dollar volume and ATR,
premarket volume and relative volume, and current bid/ask spread, then ranks the remaining names.
Liquidity thresholds and basket size are configurable through `.env.example`.

The daily optimizer downloads recent 1-minute history for the top three selected symbols and runs
the walk-forward and purged combinatorial validation engines. It records selected parameters,
PBO, DSR, and OOS diagnostics in
`logs/daily_profile.json`, but never applies a newly optimized profile automatically. Candidate
variants are recorded in `logs/strategy_trial_ledger.json` and must pass a multiple-testing-adjusted
Sharpe hurdle, PBO below 0.20, DSR probability of at least 0.95, the statistical gates, and a
final blind holdout before being written to
`data/staging/`. The isolated holdout is read only by its one-candidate validator after parameters
are frozen; it is never passed into walk-forward optimization. Existing `data/*.csv` history is not
considered blind and will not be used as a substitute. Until genuinely untouched files are supplied
as `data/final_holdout/SPY.csv` (and corresponding symbol files), candidate audits fail closed.

Audit details, holdout status, cumulative variant count, and simulated friction are written to
`logs/strategy_audit_report.json` and shown on the local dashboard. Forward-paper expectancy is
reported as unavailable until actual forward-paper tracking exists. To approve a staged candidate
for paper use, review its audit and then run:

```powershell
python -m server.evolution --approve equity-daily-YYYYMMDD-a1b2c3d4
```

Replace the example with the exact `candidate_id` from the audit report, then type that ID when
prompted. This records approval in
`data/active_paper_candidate.json`; restart the runner to load the approved parameters. Approval
does not enable live trading. A given holdout file set is evaluated at most once; a later candidate
requires a genuinely new, untouched holdout dataset. Backtest costs default to a 1 bp minimum slippage, an additional
0.1-ATR slippage component, $0.0035/share commission with a $0.35 minimum, and 0.3 bps sell-side
fees. CLI validation additionally charges a 2 bps quoted spread (half-spread on each fill).
Override with `--spread-bps` (or zero to omit quoted spread). Override other costs via
`BACKTEST_SLIPPAGE_BPS`, `BACKTEST_SLIPPAGE_ATR_MULT`,
`BACKTEST_COMMISSION_PER_SHARE`, `BACKTEST_COMMISSION_MIN`, and `BACKTEST_SELL_FEE_BPS`.
The default `alpha` strategy remains unchanged. `--strategy orb` evaluates a separate
research-only 15-minute opening-range breakout signal on the requested equities; it is not
wired into the autonomous runner or any order-routing path. CPCV and walk-forward parameter
replays include its two-value reward/risk grid and cumulative trial count. The volume and
volatility gates and opening-range-width trail are fixed research features, not optimizable
parameters in this deliberately coarsened search. The intraday ATR baseline is the prior ten
valid sessions' 14-bar ATR at the same minute of the session; incomplete opening ranges and
missing bars invalidate the corresponding observations.
`server.portfolio_allocation.build_equal_risk_plan` produces a static capped equal-risk report
for strategy sleeves and their assets (default: SPY, QQQ, NVDA). A sleeve receives a research
budget only after CPCV PBO/DSR, positive walk-forward expectancy and P&L after costs, locked
holdout, and cost-calibration gates pass; otherwise its budget is zero. The portfolio cap defaults
to 0.5% total risk (0.25% per sleeve for two sleeves, evenly
split across that sleeve's assets). The report always sets execution risk to zero, keeps dynamic
weights disabled, and cannot authorize orders.
Volume delta is approximated from OHLCV bars; live order-book imbalance is unavailable through this
polling feed.

The runner uses fixed-fractional risk sizing (clamped to at most 1% equity per trade), a hard
2% session drawdown circuit breaker, quote-aware marketable-limit bracket entries, and split
1.5R/runner profit-taking brackets when size permits. Stops move monotonically to breakeven and
then trail by Chandelier ATR. NY session phases gate signals: the opening window requires an
opening-range breakout or same-time historical RVOL momentum in addition to the existing alpha
confluence; the 12:00–14:00 lull requires a higher score and VWAP mean-reversion confirmation;
power-hour entries require daily trend alignment and ADX confirmation. New entries stop at 15:30
ET, and remaining positions are flattened from 15:50 ET. ORB/RVOL are eligibility overlays on
the current alpha engine, not standalone strategies or optimized profiles.

The runner also uses exponential reconnect backoff, single-instance locking, and structured
rotating logs at `logs/autonomous.log`. Each evaluated setup logs RVOL, ADX, ATR, score,
risk/reward, and acceptance/rejection reason; the latest scan is included in
`logs/runner_heartbeat.json` for dashboard display. The heartbeat refresh thread updates every
15 seconds, including during long synchronous optimization/API calls and market-closed waits.
The read-only Streamlit monitor consumes `logs/market_dashboard_snapshot.json` for SPY/QQQ/NVDA
quote cards, intraday closes, ATR/BBWP, and 50-day trend; it never fetches market data itself.
Its CPCV page displays persisted split results and an interactive threshold comparison that does
not rerun validation or alter promotion gates. The Local RAG assistant retrieves from mounted
Python source, allowlisted reports, strategy files, and bounded log tails. To enable local model
generation, set `APEX_UI_OLLAMA_URL` to an Ollama endpoint reachable only on the private host or
Docker network and optionally set `APEX_UI_OLLAMA_MODEL` (default
`qwen2.5-coder:1.5b`). The dashboard rejects public Ollama URLs and falls back to local evidence
excerpts when generation is not configured or unavailable. No paid API or embedding service is
used. Crypto market cards appear only when `CRYPTO_SYMBOLS` is explicitly configured.
Entries additionally require daily and hourly SMA20/50/200 alignment and ADX strength, with
completed 5-minute and 15-minute trend confirmation. Missing higher-timeframe history rejects
the signal instead of silently bypassing the filter. The research state publishes matching daily
and hourly SMA/ADX diagnostics. The local calendar at
`data/macro_events.json` accepts high-impact events with explicit UTC/offset timestamps, for
example `{"events":[{"name":"CPI","impact":"high","timestamp":"2026-11-12T13:30:00Z"}]}`;
entries pause 15 minutes before and after each event. A 1% SPY move across the trailing 16
one-minute bars also triggers the volatility circuit breaker. Keep the calendar updated; no
third-party event feed is queried.

Set `CRYPTO_SYMBOLS=BTC/USD,ETH/USD` to monitor Alpaca crypto data continuously, including
outside US equity hours. These pairs are research-only and are not submitted as orders because
the current execution path cannot guarantee broker-native OCO/bracket protection for crypto.
Restart any runner process started before this heartbeat update so it loads the new code. It will
close only a position uniquely matched to a filled order whose
client ID starts with `apex-spy-paper-test-`; any other pre-existing exposure in the selected or
configured symbols blocks startup instead of being liquidated automatically.

Check connectivity without submitting orders:

```powershell
python -m server.autonomous_runner --dry-run
```

Start the continuous paper session from the project root:

```powershell
python -m server.autonomous_runner
```

Or double-click `start_trading_session.bat`. Stop the process with Ctrl+C. The runner follows
Alpaca's exchange clock rather than assuming fixed Israel-time hours; regular US market hours
shift with daylight-saving transitions.

To register an unattended, hidden Windows Task Scheduler task (current Windows user must be
logged in; no password is stored), run PowerShell as that user from the project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_trading_task.ps1
```

It adds startup, logon, and weekday 2:00 PM local-time triggers for the unified supervisor.
Confirm Windows is set to the intended local time zone and use Task Scheduler to review or remove
the `ApexQuant-Paper-Trading` task. The hidden WSH launcher starts `start_unified.bat`; monitor
`logs/autonomous.log` and the loopback dashboard. The runner remains PAPER-only and should first
be exercised with `python -m server.autonomous_runner --dry-run`; it does not
guarantee profitable execution. The dry run exits with status 2 when the API works but a
configured/dynamically selected symbol still has an open position or order.

## Local monitoring dashboard

Start the read-only dashboard in a separate terminal or by double-clicking
`start_dashboard.bat`:

```powershell
python -m server.dashboard
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). It binds to loopback only, polls the
runner heartbeat, profile, local activity log, and Alpaca paper account every three seconds, and
continues to show local status if the API, profile, or log is unavailable. The dashboard never
submits or modifies orders. The dashboard is same-origin only and does not enable cross-origin
browser requests. No additional web packages or EXE build are required. This local dashboard is
not the blueprint's Streamlit deployment and is not configured for public access; keep it on
loopback until the authenticated reverse-proxy setup is completed.

## Production dashboard deployment

The full read-only Streamlit operational dashboard is [`app.py`](./app.py). It displays runner
health and signal telemetry, account/position/order snapshots, research progress, knowledge-base
counts, validation/audit metrics, staged candidates, macro events, and recent execution logs.
Account values are published through a sanitized snapshot by the paper runner; broker credentials
are not passed into the dashboard container. The deployment sequence for OAuth2 Proxy, Nginx,
Let's Encrypt, persistent storage, read-only source mounts, and VPS verification is documented in
[`deploy/phase1/README.md`](./deploy/phase1/README.md). The base isolated-network topology is in
[`deploy/phase0/README.md`](./deploy/phase0/README.md). ChromaDB remains isolated and must not be
enabled until its deployed runtime is verified safe for CVE-2026-45829.

## Unified paper supervisor

Set `TRADING_MODE=PAPER`, `BROKER=ALPACA`, `ALPACA_API_KEY`, and `ALPACA_API_SECRET` in `.env`,
then launch the single-process supervisor from the project root:

```powershell
python -m server.unified_master
```

Or double-click `start_unified.bat`. It starts the existing paper-only runner, the loopback-only
read-only dashboard at [http://127.0.0.1:5000](http://127.0.0.1:5000), an advisory regime/research
worker, an arsenal statistics evaluator, and a failure watchdog. The researcher runs independently
of equity market hours, refreshing regime context and checking local equity CSV history for new
data every 15 minutes. New datasets receive a bounded, cost-adjusted walk-forward discovery cycle;
repeated dataset fingerprints are not recounted as new trials. Research progress is written to
`logs/research_agent_state.json`, `logs/strategy_trial_ledger.json`, and the
`research_activity` section of `logs/strategy_audit_report.json`. Ctrl+C requests a coordinated
shutdown. The dashboard reports runner and researcher status separately; all components continue
to write logs to `logs/autonomous.log`.

The supervisor also runs a local research ingestion worker. It reads arXiv quantitative-finance
RSS feeds by default; optional public RSS URLs (including GitHub repository release feeds) can be
added with `APEX_RESEARCH_FEEDS`. Optional YouTube transcript extraction uses the local `yt-dlp`
executable and URLs listed in `APEX_RESEARCH_YOUTUBE_URLS`; transcripts are processed transiently
and are not stored. With `GEMINI_API_KEY` set, the worker uses Gemini's JSON response schema and
then applies the same strict local validation. Without a key, it falls back to local Ollama at
`127.0.0.1`; Gemini request failures do not silently switch providers. The SQLite knowledge
repository is `data/knowledge_brain.sqlite`; progress is exposed in `/api/state` as
`research_brain` and `logs/research_brain_state.json`. `INGESTION_SCHEDULE_MINUTES` defaults to
180 minutes; `CLOUD_DEPLOYMENT` marks deployment telemetry and does not change order permissions.

Strategies with extraction conviction >= 0.80 are emitted as literal-only Python research
artifacts under `data/knowledge_variants/`. These artifacts are untested specifications, are never
imported or executed by the runner, and do not become trading signals. The current backtester does
not implement arbitrary extracted entry/exit conditions, so the ingestion worker does not claim
that these are simulated or promoted. The existing bounded walk-forward researcher continues to
test its explicitly supported parameter grid independently. Alpaca crypto pairs can be monitored
for research around the clock when configured, but crypto order routing remains disabled until
protective exits are verified; Alpaca does not provide the requested Forex execution through this
connector. No generated strategy can bypass paper-only routing, statistical/holdout gates, or
human candidate review.

Knowledge Brain extraction additionally rejects strategies without numerical indicator periods,
entry and exit thresholds, and a logical-rigor score of at least 0.80. Asset routing eligibility
is a separate fail-closed metrics check: at least 20 recorded simulated trades, win rate >= 55%,
profit factor >= 1.50, logical rigor >= 0.80, and combined conviction >= 0.80. Simulation metrics
must be recorded explicitly through the Knowledge Brain API; extracted web content cannot create
them. A candidate can be promoted only after its individual simulation metrics pass these gates.
The equity runner additionally requires `ENABLE_ORDER_ROUTING=true` and a promoted stock candidate;
the default is false. This gate still controls the existing supported paper alpha, it does not
compile or execute the harvested rule text as a strategy. Do not enable it as a claim that a
harvested strategy is now deployed.

For an Oracle Linux/Ubuntu host, install dependencies into the service's Python environment,
configure `.env` with file permissions restricted to the service account, then install and enable
the scheduled services and timers:

```sh
/home/ubuntu/miniforge3/envs/apex_env/bin/python -m pip install -r requirements.txt
sudo cp apex_quant.service /etc/systemd/system/
sudo cp apex_quant.timer /etc/systemd/system/
sudo cp apex_ingestion.service /etc/systemd/system/
sudo cp apex_ingestion.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl disable --now apex_quant.service 2>/dev/null || true
sudo systemctl reset-failed apex_quant.service
sudo systemctl enable --now apex_quant.timer
sudo systemctl enable --now apex_ingestion.timer
sudo systemctl list-timers apex_quant.timer
sudo systemctl list-timers apex_ingestion.timer
sudo systemctl start apex_quant.service  # optional immediate verification
sudo systemctl status apex_quant.service
sudo journalctl -u apex_quant.service -n 100 --no-pager
sudo journalctl -u apex_ingestion.service -n 100 --no-pager
```

When updating an existing VPS deployment, make sure the application files are synchronized too:
the host must have `server/ingestion/cli.py`, the current `config/settings.py`, and all package
modules from this checkout. `systemctl daemon-reload` only reloads unit definitions; it does not
deploy Python source files. The ingestion unit sets `PYTHONPATH` explicitly and checks that its
entrypoint exists before launch. After syncing code, verify imports before starting the timer:

```sh
cd /home/ubuntu/apex_quant
test -f server/ingestion/cli.py
/home/ubuntu/miniforge3/envs/apex_env/bin/python -c \
  "from config.settings import load_project_env; from server.ingestion.cli import main; print('ingestion imports OK')"
sudo systemctl disable --now apex_ingestion.timer
sudo cp apex_ingestion.service /etc/systemd/system/
sudo cp apex_ingestion.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl reset-failed apex_ingestion.service
sudo systemctl enable --now apex_ingestion.timer
sudo systemctl start apex_ingestion.service
sudo journalctl -u apex_ingestion.service -n 100 --no-pager
```

The service uses the Python 3.10 Miniforge environment at
`/home/ubuntu/miniforge3/envs/apex_env`. Before enabling the backtest timer, copy the validated
`data/SPY.csv`, `data/QQQ.csv`, and `data/NVDA.csv` files to the cloud host and verify they are
present. The oneshot runs daily at midnight UTC, performs walk-forward-only evaluation, and logs
to journald; it does not fetch live data or submit orders. A missing/empty/invalid dataset returns
a failure rather than a successful empty evaluation.

Both systemd units require `/home/ubuntu/apex_quant/.env` via `EnvironmentFile=` and application
entrypoints independently load the project-root file through `python-dotenv`. This makes
interactive/Windows launches independent of their current directory while systemd fails clearly
if its configured environment file is missing. Keep both Alpaca variables named exactly
`ALPACA_API_KEY` and `ALPACA_API_SECRET`; do not place their values in logs or source control.

The separate research-ingestion oneshot runs 10 minutes after boot and then every 180 minutes.
It reads `GEMINI_API_KEY` from `/home/ubuntu/apex_quant/.env`, which must be configured on the host
without committing the secret:

```sh
sudoedit /home/ubuntu/apex_quant/.env
sudo chmod 600 /home/ubuntu/apex_quant/.env
sudo chown ubuntu:ubuntu /home/ubuntu/apex_quant/.env
```

Add a `GEMINI_API_KEY=...` line in the editor, then install `google-genai` into `apex_env` as
shown above. Gemini ingestion is independent of the backtest job. Without the key, extraction
falls back to local Ollama; the ingestion service logs a warning about this. The backtest
workload does not call Alpaca or submit orders. `ENABLE_ORDER_ROUTING` remains false.

The CLI reports trade count, win rate, expectancy in R, gross P&L, transaction costs, net P&L,
profit factor, drawdown, and out-of-sample walk-forward metrics. A prior 160-day single-symbol
synthetic verification completed with 24 strategy trades (25.0% wins, -0.411R expectancy,
-$8.45 gross, $20.38 costs, -$28.83 net) versus 61 pooled null-model trades (34.4% wins,
-0.242R expectancy, +$25.59 gross, $50.32 costs, -$24.73 net). This intentionally no-edge
synthetic test demonstrates costs and reporting, not strategy profitability or live performance.

Check the paper API and selected-symbol exposure without starting services or submitting orders:

```powershell
python -m server.unified_master --dry-run
```

Research and arsenal evolution do not submit orders or change the runner's selected strategy.
Regime classification currently uses daily US-equity bars. Crypto/forex and catalogued ORB/VWAP/
RVOL strategies are research-only and are not represented as supported live execution. Strategies
remain `research_only` until recorded paper statistics meet Sharpe > 1.5, profit factor > 1.3,
drawdown < 5%, and at least 30 paper trades; three consecutive negative R outcomes or non-positive
recent expectancy against a positive baseline quarantines a strategy. Populate these metrics from
validated paper results before relying on promotion status.

The watchdog diagnoses and logs component exceptions and retries failed components with
exponential backoff. It deliberately does not edit source files, modify risk controls, or roll back
the workspace automatically: runtime self-patching and destructive Git rollback are disabled.
#   a p e x - q u a n t  
 