# Wikispooks Pipeline

Wikispooks articles (wiki-style deep-research pages). Text extraction via Firecrawl with DDOS bypass URL.

## DDOS Bypass

Wikispooks serves a DDOS protection wall on direct article URLs. The bypass is a one-click human verification URL:

```
https://wikispooks.com/w/welcome_back.php?goto=%2Fwiki%2FARTICLE%2FPATH
```

**URL transformation rule:** replace `/wiki/` prefix with `/w/welcome_back.php?goto=`, and URL-encode the path:
- `/wiki/COVID-19/Perpetrators/Bilderberg` → `/w/welcome_back.php?goto=%2Fwiki%2FCOVID-19%2FPerpetrators%2FBilderberg`

Both `web_extract` and Firecrawl work with the bypass URL. Never use the direct URL — it returns the DDOS page, not the article.

## Phase 0 — Dedup + Firecrawl health check

```bash
# Dedup (check both direct URL and bypass URL)
if grep -rql "source_url: $DIRECT_URL\|$BYPASS_URL" "$OBSIDIAN_VAULT_PATH/Knowledge base/" 2>/dev/null; then
    echo "Already in vault — skip"
    exit 0
fi

# Firecrawl health
curl -s --max-time 5 http://localhost:3002/v2/scrape \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com","formats":["markdown"]}' > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Firecrawl down — skip"
    exit 0
fi
```

## Planner — Ticket creation

```bash
hermes kanban --board knowledge-base create \
  --assignee researcher \
  --skill knowledge-base \
  --max-runtime 3600 \
  --body "Wikispooks article — <title>. Deep treatment required.

Use BYPASS URL: https://wikispooks.com/w/welcome_back.php?goto=<encoded_path>
Do NOT use the direct URL — it returns a DDOS protection wall.

Firecrawl extraction → read ENTIRE source → Key Claims (≥4, with direct quotes) → Section-by-section analysis (dominant section) → Context → Critical Analysis → Nuances. Target 15K-25K chars. Upload raw markdown to MinIO (knowledge-base/articles/). Add source_files: field with Tailscale FQDN.

1. <DIRECT_URL> (use bypass: <BYPASS_URL>)

Langue: contenu en langue source, labels en anglais. Save to Knowledge base/. Push après la note." \
  "KB: Wikispooks — <title>"
```

## Researcher worker

### Phase 1 — Extraction

```bash
# Transform URL to bypass
DIRECT_URL="<url>"
ARTICLE_PATH=$(echo "$DIRECT_URL" | sed 's|https://wikispooks.com/wiki/||')
ENCODED_PATH=$(python3 -c "import urllib.parse; print(urllib.parse.quote('/wiki/$ARTICLE_PATH', safe=''))")
BYPASS_URL="https://wikispooks.com/w/welcome_back.php?goto=$ENCODED_PATH"

# Extract via Firecrawl
curl -s --max-time 60 http://localhost:3002/v2/scrape \
  -H "Content-Type: application/json" \
  -d "{\"url\":\"$BYPASS_URL\",\"formats\":[\"markdown\"]}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['markdown'])" \
  > /tmp/wikispooks_raw.md
```

### Phase 2 — Metadata

| Field | Source |
|-------|--------|
| `title` | First `#` in markdown or `<title>` tag |
| `source` | `Wikispooks — <Page Title>` |
| `source_url` | **Direct URL** (not bypass) — e.g. `https://wikispooks.com/wiki/COVID-19/Perpetrators/Bilderberg` |
| `date` | Page last-modified (if available), else YYYY of content |

### Phase 3 — Deep analysis

Follow the deep treatment standard from `knowledge-base` SKILL.md (Template section). No light template. Required depth:

- **Read the ENTIRE source** — use read_file with offset/limit. Never rely on web_extract summaries. Read completely before writing.
- **Key Claims** — ≥4 specific claims with direct quotes (`>`) and analysis
- **Section-by-section analysis** — dominant section, longer than Key Claims + Critical Analysis combined. Wikispooks pages have structured sections — cover each one.
- **Context** — Wikispooks is a deep politics wiki with a specific editorial stance. Note this.
- **Critical Analysis** — what's documented vs speculative, sourcing quality, omissions
- **Nuances** — limitations, editorial bias, gaps
- **Size target:** 15 000-25 000 chars
- **See Also** — only existing vault notes (grep before linking)

### Phase 4 — MinIO archive

```bash
SLUG="wikispooks-<page-slug>"
mc cp /tmp/wikispooks_raw.md "minio/knowledge-base/articles/${SLUG}.md"

# Verify
mc ls "minio/knowledge-base/articles/${SLUG}.md"
```

### Phase 5 — Obsidian note

```yaml
source_files:
  text: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/articles/wikispooks-<slug>.md
```

### Phase 6 — Git push

```bash
cd "$OBSIDIAN_VAULT_PATH"
git add "Knowledge base/${SLUG}.md"
git commit -m "add: ${SLUG}"
git push
```
