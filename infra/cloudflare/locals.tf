locals {
  spec      = yamldecode(file("${path.module}/spec.yaml"))
  consumers = yamldecode(file("${path.module}/consumers.yaml")).consumers

  zone_name     = local.spec.zone_name
  tunnel_name   = local.spec.tunnel_name
  mcp_hostname  = local.spec.mcp.hostname
  mcp_origin    = local.spec.mcp.origin
  ingress_rules = local.spec.tunnel_ingress

  consumer_precedence = {
    for idx, key in local.spec.consumer_keys : key => idx + 1
  }
}
