from pathlib import Path


def test_github_oauth_has_required_scope_and_exact_redirect_allowlist():
    compose = (
        Path(__file__).parents[1] / "deploy" / "phase0" / "docker-compose.yml"
    ).read_text(encoding="utf-8")

    assert "--scope=user:email read:org" in compose
    assert "--authenticated-emails-file=/etc/oauth2-proxy/allowed_emails" in compose
    assert "--redirect-url=https://151.145.93.209.nip.io/oauth2/callback" in compose
    assert "--whitelist-domain=151.145.93.209.nip.io" in compose
