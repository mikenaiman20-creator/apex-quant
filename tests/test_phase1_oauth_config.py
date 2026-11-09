from pathlib import Path


def test_github_oauth_has_required_scope_and_exact_redirect_allowlist():
    compose = (
        Path(__file__).parents[1] / "deploy" / "phase0" / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    env_example = (
        Path(__file__).parents[1] / "deploy" / "phase0" / "oauth2-proxy.env.example"
    ).read_text(encoding="utf-8")

    assert "--scope=user:email read:org" in compose
    assert "--authenticated-emails-file=/etc/oauth2-proxy/allowed_emails" in compose
    # Every login attempt needs its own CSRF cookie. With the v7 default
    # (false) a retry or a second tab overwrites the cookie and the callback
    # fails with "Unable to find a valid CSRF token".
    assert "--cookie-csrf-per-request=true" in compose
    # Domain-specific flags moved to the environment file so local and staging
    # deployments cannot drift between http:// and https:// callbacks.
    assert "--redirect-url" not in compose
    assert "--whitelist-domain" not in compose
    assert (
        "OAUTH2_PROXY_REDIRECT_URL=https://151.145.93.209.nip.io/oauth2/callback"
        in env_example
    )
    assert "OAUTH2_PROXY_WHITELIST_DOMAINS=.151.145.93.209.nip.io" in env_example
    assert "OAUTH2_PROXY_COOKIE_SECURE=true" in env_example


def test_phase1_nginx_has_one_tls_server_and_resolvable_upstreams():
    nginx = (
        Path(__file__).parents[1] / "deploy" / "phase1" / "nginx.conf"
    ).read_text(encoding="utf-8")

    # A duplicate 443 server block was once appended by hand and pointed at the
    # non-existent "streamlit-1" host, crash-looping Nginx at startup.
    assert nginx.count("listen 443 ssl") == 1
    assert "proxy_pass http://streamlit-1" not in nginx
    assert "proxy_pass http://apex-project-streamlit-1:8501" in nginx
    assert "proxy_pass http://apex-project-oauth2-proxy-1:4180" in nginx

    # The OAuth endpoints must bypass auth_request or the login flow recurses.
    oauth_block = nginx.split("location /oauth2/ {", 1)[1].split("}", 1)[0]
    assert "auth_request" not in oauth_block
    assert "proxy_set_header Host $http_host;" in oauth_block
    assert "proxy_set_header X-Forwarded-Proto $scheme;" in oauth_block

