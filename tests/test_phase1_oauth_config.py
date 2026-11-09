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
