variable "cloudflare_api_token" {
  type        = string
  sensitive   = true
  description = "API token with Tunnel, DNS, Access, and Service Token permissions."
}

variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account ID (Zero Trust + tunnel owner)."
}

variable "cloudflare_team_name" {
  type        = string
  description = "Zero Trust team subdomain (e.g. sevend for sevend.cloudflareaccess.com)."
}
