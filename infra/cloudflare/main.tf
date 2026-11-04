data "cloudflare_zone" "sevend" {
  filter = {
    name = local.zone_name
  }
}

data "cloudflare_zero_trust_tunnel_cloudflared" "sevend_io_stream" {
  account_id = var.cloudflare_account_id
  filter = {
    name = local.tunnel_name
  }
}

resource "cloudflare_dns_record" "mcp" {
  zone_id = data.cloudflare_zone.sevend.id
  name    = "mcp"
  content = "${data.cloudflare_zero_trust_tunnel_cloudflared.sevend_io_stream.id}.cfargotunnel.com"
  type    = "CNAME"
  ttl     = 1
  proxied = true
}

resource "cloudflare_zero_trust_access_application" "mcp" {
  account_id                = var.cloudflare_account_id
  name                      = "KB Agent MCP (${local.mcp_hostname})"
  domain                    = local.mcp_hostname
  type                      = "self_hosted"
  session_duration          = "24h"
  skip_interstitial         = true
  auto_redirect_to_identity = false

  policies = [
    for key in local.spec.consumer_keys : {
      id         = cloudflare_zero_trust_access_policy.mcp_consumer[key].id
      precedence = local.consumer_precedence[key]
    }
  ]
}

resource "cloudflare_zero_trust_tunnel_cloudflared_config" "sevend_io_stream" {
  account_id = var.cloudflare_account_id
  tunnel_id  = data.cloudflare_zero_trust_tunnel_cloudflared.sevend_io_stream.id

  config = {
    ingress = [
      for rule in local.ingress_rules : merge(
        {
          service = rule.service
        },
        lookup(rule, "hostname", null) != null ? { hostname = rule.hostname } : {},
        lookup(rule, "hostname", null) == local.mcp_hostname ? {
          origin_request = {
            access = {
              required  = true
              team_name = var.cloudflare_team_name
              aud_tag   = [cloudflare_zero_trust_access_application.mcp.aud]
            }
          }
        } : {}
      )
    ]
  }
}
