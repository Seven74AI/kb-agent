output "mcp_hostname" {
  value       = local.mcp_hostname
  description = "Public MCP hostname secured by Cloudflare Access."
}

output "mcp_url" {
  value       = "https://${local.mcp_hostname}/mcp"
  description = "Streamable HTTP MCP endpoint (requires CF-Access-Client-Id/Secret)."
}

output "tunnel_id" {
  value       = data.cloudflare_zero_trust_tunnel_cloudflared.sevend_io_stream.id
  description = "sevend-io-stream tunnel UUID."
}

output "access_application_aud" {
  value       = cloudflare_zero_trust_access_application.mcp.aud
  description = "Access application audience tag (tunnel origin_request)."
}

output "service_token_client_ids" {
  value = {
    for key, token in cloudflare_zero_trust_access_service_token.consumer :
    key => token.client_id
  }
  description = "Per-consumer CF-Access-Client-Id values (store in client secret managers)."
}

output "service_token_client_secrets" {
  value = {
    for key, token in cloudflare_zero_trust_access_service_token.consumer :
    key => token.client_secret
  }
  sensitive   = true
  description = "Per-consumer CF-Access-Client-Secret values (shown once at create/rotate)."
}
