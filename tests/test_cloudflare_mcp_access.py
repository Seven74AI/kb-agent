"""Cloudflare Access + tunnel spec for mcp.sevend.io (infra/cloudflare)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

INFRA_DIR = Path(__file__).resolve().parents[1] / "infra" / "cloudflare"
SPEC_PATH = INFRA_DIR / "spec.yaml"
CONSUMERS_PATH = INFRA_DIR / "consumers.yaml"


def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def spec() -> dict:
    return _load_yaml(SPEC_PATH)


@pytest.fixture
def consumers_doc() -> dict:
    return _load_yaml(CONSUMERS_PATH)


def test_spec_declares_mcp_hostname_and_tunnel(spec: dict):
    assert spec["tunnel_name"] == "sevend-io-stream"
    mcp = spec["mcp"]
    assert mcp["hostname"] == "mcp.sevend.io"
    assert mcp["origin"].startswith("http://")
    assert "8002" in mcp["origin"]


def test_spec_ingress_includes_mcp_and_catch_all(spec: dict):
    hostnames = [r.get("hostname") for r in spec["tunnel_ingress"]]
    assert "mcp.sevend.io" in hostnames
    services = [r["service"] for r in spec["tunnel_ingress"]]
    assert any(s == "http_status:404" for s in services)
    assert spec["tunnel_ingress"][-1]["service"] == "http_status:404"


def test_mcp_ingress_routes_to_local_mcp_server(spec: dict):
    mcp_rules = [
        r
        for r in spec["tunnel_ingress"]
        if r.get("hostname") == "mcp.sevend.io"
    ]
    assert len(mcp_rules) == 1
    assert mcp_rules[0]["service"] == spec["mcp"]["origin"]


def test_consumers_are_unique_and_non_empty(consumers_doc: dict):
    consumers = consumers_doc["consumers"]
    assert len(consumers) >= 2
    keys = list(consumers)
    assert len(keys) == len(set(keys))
    for key, meta in consumers.items():
        assert key.replace("_", "").isalnum()
        assert meta.get("display_name", "").strip()


def test_consumers_match_spec_consumer_keys(spec: dict, consumers_doc: dict):
    spec_keys = set(spec.get("consumer_keys", []))
    file_keys = set(consumers_doc["consumers"])
    assert spec_keys == file_keys


def test_terraform_files_reference_mcp_hostname():
    locals_tf = (INFRA_DIR / "locals.tf").read_text(encoding="utf-8")
    main_tf = (INFRA_DIR / "main.tf").read_text(encoding="utf-8")
    tokens_tf = (INFRA_DIR / "service_tokens.tf").read_text(encoding="utf-8")
    assert "spec.yaml" in locals_tf
    assert "sevend-io-stream" in main_tf or "tunnel_name" in locals_tf
    assert "non_identity" in tokens_tf
    assert "cloudflare_zero_trust_access_service_token" in tokens_tf


def test_terraform_declares_per_consumer_service_tokens():
    tokens_tf = (INFRA_DIR / "service_tokens.tf").read_text(encoding="utf-8")
    assert "for_each" in tokens_tf
    assert "consumers.yaml" in tokens_tf or "local.consumers" in tokens_tf


@pytest.mark.parametrize(
    "filename",
    ["versions.tf", "variables.tf", "main.tf", "service_tokens.tf", "outputs.tf"],
)
def test_required_terraform_files_exist(filename: str):
    assert (INFRA_DIR / filename).is_file()
