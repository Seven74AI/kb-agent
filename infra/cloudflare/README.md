# Cloudflare Access for KB Agent MCP (`mcp.sevend.io`)

Secures the stateless MCP tracer (`mcp_server.py` on port **8000**, path `/mcp`) behind the existing **`sevend-io-stream`** Cloudflare Tunnel and a dedicated **Access** application.

## What this stack does

| Layer | Behavior |
| --- | --- |
| **DNS** | `mcp.sevend.io` → CNAME to `{tunnel_id}.cfargotunnel.com` (proxied) |
| **Tunnel** | Ingress on `mcp.sevend.io` → `http://127.0.0.1:8000` with Access JWT validation at the connector |
| **Access** | Self-hosted app on `mcp.sevend.io`; **Service Auth** (`non_identity`) policies — one **service token** per consumer |
| **Clients** | Send `CF-Access-Client-Id` and `CF-Access-Client-Secret` on every MCP HTTP request |

Requests **without** valid service token headers are rejected by Cloudflare Access (typically **403** or redirect to the Access login / interstitial — not the origin MCP server).

## Adopting an existing tunnel

If `sevend-io-stream` already serves other hostnames, copy those ingress rules into `spec.yaml` → `tunnel_ingress` **above** the `mcp.sevend.io` rule. Keep `http_status:404` last. This module replaces the **entire** remotely-managed tunnel config on apply.

## Apply (operator)

```bash
cd infra/cloudflare
export TF_VAR_cloudflare_api_token="..."
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

Retrieve credentials after apply:

```bash
terraform output -json service_token_client_ids
terraform output -json service_token_client_secrets
```

Store secrets per consumer (Cursor cloud vs Mac) so one can be rotated/revoked without affecting the other.

## Smoke test

With the MCP server and `cloudflared` running on the VPS:

```bash
# Without token — expect 403 (or Access redirect), not MCP JSON
curl -sS -o /dev/null -w "%{http_code}\n" "https://mcp.sevend.io/mcp"

# With token — expect 4xx/405 from MCP (not Access), proving the request reached origin
curl -sS -o /dev/null -w "%{http_code}\n" \
  -H "CF-Access-Client-Id: $CLIENT_ID" \
  -H "CF-Access-Client-Secret: $CLIENT_SECRET" \
  "https://mcp.sevend.io/mcp"
```

## Cursor MCP client

Point the remote MCP URL at `https://mcp.sevend.io/mcp` and configure the Access service token headers for that server entry (per-consumer credentials from Terraform outputs).
