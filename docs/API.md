# API reference

## Why this document matters

The FastAPI service auto-generates Swagger at `/docs` when it's running, but that's only available to whoever has the service up. This page gives client developers a complete contract to integrate against — endpoints, schemas, error shapes, examples in three languages — without booting the service first.

## Base URL

- Local: `http://localhost:8000`
- HuggingFace Spaces: usually the Gradio UI only; if you also host the FastAPI service, point clients at its public hostname.

## Authentication

None today. The endpoints are open by default. CORS is enabled for all origins ([`api.py:31-37`](../api.py#L31)).

If you need auth, add a FastAPI dependency that checks an `Authorization: Bearer <token>` header against a value pulled from an env var. The architecture supports it cleanly because the existing endpoints don't depend on per-user state.

## Endpoints

### `GET /` — health check

Returns service metadata. Use this in load balancer / Kubernetes liveness probes.

**Response 200**
```json
{
  "status": "ok",
  "model": "multilingual-fake-news-detector",
  "languages": ["English", "Urdu", "Spanish"]
}
```

**Examples**
```bash
curl http://localhost:8000/
```

```python
import requests
print(requests.get("http://localhost:8000/").json())
```

```javascript
const r = await fetch("http://localhost:8000/");
console.log(await r.json());
```

---

### `POST /predict` — single-text prediction

Classify one piece of text.

**Request body**
| Field | Type | Constraints |
|---|---|---|
| `text` | string | required, 1 ≤ length ≤ 1000, non-whitespace |
| `language` | string | required, one of `"English"`, `"Urdu"`, `"Spanish"` |

```json
{
  "text": "Scientists confirm the moon is made of cheese.",
  "language": "English"
}
```

**Response 200**
| Field | Type | Description |
|---|---|---|
| `prediction` | string | `"Real"` or `"Fake"` |
| `confidence` | float | Score for the predicted class, 0.0–1.0 |
| `scores` | object | Per-class scores, summing to ~1.0 |
| `language` | string | Echoes the input language |

```json
{
  "prediction": "Fake",
  "confidence": 0.95,
  "scores": {"Real": 0.05, "Fake": 0.95},
  "language": "English"
}
```

**Errors**
| Status | When |
|---|---|
| 422 | `text` empty, too long, blank, or `language` not in the supported set |
| 500 | Inference failure (model not loaded, downstream exception) |

**Examples**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Scientists confirm the moon is made of cheese.","language":"English"}'
```

```python
import requests
r = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Scientists confirm the moon is made of cheese.", "language": "English"},
)
print(r.json())
```

```javascript
const r = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Scientists confirm...", language: "English"})
});
console.log(await r.json());
```

---

### `POST /predict/batch` — batch prediction

Classify many texts of the same language in one request. Same model logic, different HTTP shape — saves round-trip overhead when scoring lots of items.

**Request body**
| Field | Type | Constraints |
|---|---|---|
| `texts` | array of strings | required, ≥ 1 item, each non-empty and ≤ 1000 chars |
| `language` | string | required, one of `"English"`, `"Urdu"`, `"Spanish"` |

```json
{
  "texts": [
    "Scientists confirm the moon is made of cheese.",
    "The Federal Reserve raised interest rates by 0.25 points."
  ],
  "language": "English"
}
```

**Response 200**
```json
{
  "language": "English",
  "predictions": [
    {
      "text": "Scientists confirm the moon is made of cheese.",
      "prediction": "Fake",
      "confidence": 0.95,
      "scores": {"Real": 0.05, "Fake": 0.95},
      "language": "English"
    },
    {
      "text": "The Federal Reserve raised interest rates by 0.25 points.",
      "prediction": "Real",
      "confidence": 0.92,
      "scores": {"Real": 0.92, "Fake": 0.08},
      "language": "English"
    }
  ]
}
```

**Errors** — same as `/predict`. If *any* item in `texts` fails validation, the whole batch is rejected with 422 and the index of the offending entry.

**Example**
```python
import requests
r = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": ["First headline...", "Second headline..."],
        "language": "Spanish",
    },
)
for p in r.json()["predictions"]:
    print(f"{p['prediction']} ({p['confidence']:.2f}) — {p['text'][:60]}")
```

## Error response shape

All error responses follow FastAPI's default JSON shape:

```json
{
  "detail": "human-readable message"
}
```

For 422 errors, FastAPI's standard validation response includes the field path so clients can show field-level errors:

```json
{
  "detail": [
    {
      "type": "string_too_long",
      "loc": ["body", "text"],
      "msg": "String should have at most 1000 characters",
      "input": "..."
    }
  ]
}
```

## CORS

Enabled for all origins, all methods, all headers ([`api.py:31-37`](../api.py#L31)). Browser clients can call the API directly. Tighten the policy in production if you're not behind a gateway that already does it.

## Rate limiting

Not implemented. If you put this in front of public traffic, add a middleware like [`slowapi`](https://github.com/laurentS/slowapi) or front it with a gateway (NGINX, Cloudflare, API Gateway).

## Interactive docs

When the service is running, hit:

- `GET /docs` — Swagger UI for trying calls in the browser.
- `GET /redoc` — Redoc-rendered reference.
- `GET /openapi.json` — raw OpenAPI 3 schema, useful for generating client SDKs.

## Client SDK generation

The OpenAPI spec at `/openapi.json` is enough to generate a typed client in most languages. Example for TypeScript:

```bash
npx @openapitools/openapi-generator-cli generate \
  -i http://localhost:8000/openapi.json \
  -g typescript-fetch \
  -o ./client-ts
```

For Python clients, `httpx` + a typed dataclass per endpoint is usually less ceremony than a generated SDK.
