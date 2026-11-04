resource "cloudflare_zero_trust_access_service_token" "consumer" {
  for_each = local.consumers

  account_id = var.cloudflare_account_id
  name       = each.value.display_name
  duration   = "8760h"

  lifecycle {
    create_before_destroy = true
  }
}

resource "cloudflare_zero_trust_access_policy" "mcp_consumer" {
  for_each = local.consumers

  account_id = var.cloudflare_account_id
  name       = "${each.value.display_name} (service auth)"
  decision   = "non_identity"

  include = [{
    service_token = {
      token_id = cloudflare_zero_trust_access_service_token.consumer[each.key].id
    }
  }]
}
