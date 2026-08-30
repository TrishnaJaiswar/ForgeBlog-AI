# Building Production‑Ready Generative AI Services: From Prompt Design to Observability

## Why Generative AI Needs a Production Lens

Large language models (LLMs) differ from classic static inference pipelines in two measurable ways. First, **latency** is a function of prompt length, model size, and token generation; a 50‑token prompt to a 175B model can take 800 ms, while a frozen image classifier returns a prediction in <20 ms. Second, **cost** accrues per token: OpenAI’s pricing charges $0.02 per 1 k input tokens and $0.06 per 1 k output tokens, so a 200‑token request may cost $0.016, whereas a one‑time model download is free after the initial compute. Production systems must budget both time and dollars per request, not just CPU/GPU cycles.

When user‑generated prompts travel to third‑party APIs, **regulatory and privacy** rules kick in. GDPR, CCPA, and industry‑specific mandates (HIPAA, PCI‑DSS) require that personally identifiable information (PII) never leave the trusted boundary without consent or masking. A common mitigation is to **scrub** or **tokenize** PII before the API call and retain the original locally for audit trails.

Prompt variability introduces a hidden source of **non‑determinism**. Small wording changes can flip model output, making reproducibility and debugging a nightmare. Without versioned prompt templates and logging of the exact string sent, you cannot reliably trace a faulty response back to its cause.

Below is a minimal working example (MWE) that calls OpenAI’s `gpt-3.5-turbo` and measures round‑trip latency:

```python
import time, os, openai

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt = "Summarize the privacy implications of sending user data to an LLM."

start = time.perf_counter()
resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=50,
)
latency_ms = (time.perf_counter() - start) * 1000

print(f"Latency: {latency_ms:.1f} ms")
print("Response:", resp.choices[0].message.content.strip())
```

**Edge cases**: network timeouts, API rate limits, and token‑limit errors should be caught (`except openai.error.*`) and retried with exponential back‑off. Logging the raw prompt, token usage (`resp.usage`), and latency enables post‑mortem analysis and cost tracking.

## Prompt Engineering as a First‑Class API Contract

### 1. JSON‑based prompt contract  
A stable contract lets clients evolve independently of the LLM backend. Use a flat JSON object with three explicit keys:

```json
{
  "system": "You are a helpful travel assistant.",
  "user": "Find me a cheap flight from NYC to Paris next week.",
  "context": {
    "locale": "en-US",
    "previous_messages": ["..."]
  }
}
```

* `system` – static instruction that sets the model’s persona.  
* `user` – the end‑user utterance; required and limited to a single string.  
* `context` – optional dictionary for locale, session history, or feature flags. Keeping the schema flat avoids version‑skew and makes forward‑compatible parsing trivial.

### 2. Validation layer with Pydantic  
Wrap the contract in a Pydantic model to enforce shape and size before any LLM call:

```python
from pydantic import BaseModel, Field, validator

MAX_TOKENS = 1024

class PromptContract(BaseModel):
    system: str = Field(..., min_length=1, max_length=200)
    user: str = Field(..., min_length=1, max_length=800)
    context: dict | None = None

    @validator("*", pre=True)
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v

    @validator("user")
    def enforce_token_budget(cls, v):
        # Rough token estimate: 1 token ≈ 4 characters
        if len(v) / 4 > MAX_TOKENS:
            raise ValueError("User prompt exceeds token budget")
        return v
```

Malformed JSON, missing fields, or prompts that would exceed the token budget raise a `ValidationError`, preventing costly API calls.

### 3. Production‑readiness checklist  
- **Rate‑limit** – enforce per‑client QPS (e.g., 5 req/s) to protect downstream LLM quotas.  
- **Sanitization** – strip PII, HTML tags, or injection patterns; why? to avoid leaking sensitive data to the model.  
- **Token‑budget enforcement** – reject or truncate prompts that would push the total request over the model’s limit; why? to guarantee predictable latency and cost.  

### 4. Code sketch: serialize, call, log usage  

```python
import json, logging, time
import openai  # assume OpenAI-compatible endpoint

log = logging.getLogger("prompt_api")
log.setLevel(logging.INFO)

def call_llm(contract: PromptContract) -> str:
    payload = contract.dict()
    # Serialize to JSON for transport
    body = json.dumps(payload)

    start = time.time()
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": payload["system"]},
            {"role": "user",   "content": payload["user"]},
        ],
        max_tokens=512,
    )
    elapsed = time.time() - start

    # Log token usage and latency
    usage = resp["usage"]
    log.info(
        "LLM call",
        extra={
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "latency_ms": int(elapsed * 1000),
        },
    )
    return resp["choices"][0]["message"]["content"]
```

**Edge cases**:  
- *ValidationError*: return HTTP 400 with error details.  
- *LLM timeout*: retry with exponential back‑off or fallback to a cached response.  
- *Unexpected token count*: log a warning and truncate the user message to stay within budget.

By treating the prompt as a versioned API contract, you isolate client changes, enforce safety checks early, and retain observability over cost and performance.

## Implementing a Scalable Inference Service

**Dockerfile – reproducible, memory‑capped, health‑checked**  
```dockerfile
# Use a lightweight Python base
FROM python:3.11-slim

# Install model SDK (e.g., HuggingFace Transformers) and FastAPI
RUN pip install --no-cache-dir \
    "transformers[torch]" \
    fastapi uvicorn[standard]

# Copy service code
COPY app/ /app/
WORKDIR /app

# Fixed memory limit (e.g., 4 GB) – enforced by the container runtime
ENV MEMORY_LIMIT=4g

# Expose health‑check endpoint for orchestration platforms
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
*Why*: A deterministic memory ceiling prevents OOM crashes during traffic spikes, keeping costs predictable.

---

**Async FastAPI route – streaming token chunks**  
```python
# main.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.post("/generate")
async def generate(request: Request):
    payload = await request.json()
    prompt = payload["prompt"]
    max_new_tokens = payload.get("max_new_tokens", 50)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    def token_stream():
        with torch.no_grad():
            for output in model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                stream=True,          # pseudo‑API; replace with actual streaming logic
            ):
                chunk = tokenizer.decode(output, skip_special_tokens=True)
                yield f"data:{chunk}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
```
*Why*: Streaming reduces perceived latency because the client receives the first tokens while the model continues generating.

---

**Benchmark script – throughput vs. batch size & $/token**  
```python
# benchmark.py
import time, json, requests

URL = "http://localhost:8000/generate"
PROMPT = "Explain quantum tunneling in one sentence."
COST_PER_1M_TOKENS = 0.02  # USD, example pricing

def run(batch):
    start = time.time()
    responses = [
        requests.post(URL, json={"prompt": PROMPT, "max_new_tokens": 30})
        for _ in range(batch)
    ]
    total_tokens = sum(len(r.text.split()) for r in responses)
    elapsed = time.time() - start
    throughput = total_tokens / elapsed
    cost = (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
    return {"batch": batch, "throughput_tps": throughput, "cost_per_token": cost / total_tokens}

if __name__ == "__main__":
    for b in [1, 4, 8, 16]:
        print(json.dumps(run(b)))
```
*Trade‑off*: Larger batches improve tokens‑per‑second but increase tail latency; pick a batch size that meets SLA while keeping $/token low.

---

**Token‑budget middleware – abort over‑budget requests**  
```python
# middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

MAX_TOKENS = 150  # configurable ceiling

class TokenBudgetMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.json()
        prompt = body.get("prompt", "")
        # Rough estimate: 1 token ≈ 4 characters
        est_tokens = len(prompt) // 4 + body.get("max_new_tokens", 0)
        if est_tokens > MAX_TOKENS:
            raise HTTPException(status_code=429,
                                detail=f"Request exceeds token budget of {MAX_TOKENS}")
        return await call_next(request)

app.add_middleware(TokenBudgetMiddleware)
```
*Edge case*: If a client sends a very long prompt that exceeds the budget, the middleware returns 429 early, avoiding wasted GPU cycles. Adjust `MAX_TOKENS` per pricing tier to keep spend predictable.

## Observability, Debugging, and Edge‑Case Handling

**1. Request/response logging with correlation IDs**  
Every inbound request receives a UUID that travels through the stack. Log the request and the model’s raw response, but strip any PII before writing to the log sink.

```python
import uuid, json, re
from structlog import get_logger

log = get_logger()

def redact_pi(text: str) -> str:
    # Simple email/phone redaction; replace with a robust library in prod
    return re.sub(r'[\w\.-]+@[\w\.-]+', '[REDACTED_EMAIL]', 
                  re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', text))

def handle(request_body: dict):
    corr_id = str(uuid.uuid4())
    log = log.bind(correlation_id=corr_id)

    safe_prompt = redact_pi(request_body["prompt"])
    log.info("request_received", prompt=safe_prompt)

    response = model.generate(request_body["prompt"])
    safe_resp = redact_pi(response.text)
    log.info("response_sent", response=safe_resp)
    return {"id": corr_id, "output": response.text}
```

*Why*: Correlation IDs let you stitch together logs across services, and redaction protects user privacy.

---

**2. Prometheus metrics**  
Expose a `/metrics` endpoint that reports:

- `genai_request_latency_seconds` (histogram)
- `genai_input_tokens_total` (counter)
- `genai_error_rate` (gauge)
- `genai_hallucination_flag_total` (counter, labeled by `model_version`)

```go
var (
    latency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
        Name: "genai_request_latency_seconds",
        Buckets: prometheus.ExponentialBuckets(0.01, 2, 10),
    }, []string{"model_version"})
    tokens = prometheus.NewCounterVec(prometheus.CounterOpts{
        Name: "genai_input_tokens_total",
    }, []string{"model_version"})
    errors = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "genai_error_rate",
    })
    hallucinations = prometheus.NewCounterVec(prometheus.CounterOpts{
        Name: "genai_hallucination_flag_total",
    }, []string{"model_version"})
)
```

*Trade‑off*: Histograms give latency distribution but increase memory; tune bucket count based on traffic volume.

---

**3. OpenTelemetry tracing span**  
Wrap each generation call in a span that records prompt size, model version, and downstream API latency.

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind

tracer = trace.get_tracer("genai-service")

def generate_with_trace(prompt: str, model: str):
    with tracer.start_as_current_span(
        "model.generate", kind=SpanKind.CLIENT,
        attributes={
            "prompt.length": len(prompt),
            "model.version": model,
        }
    ) as span:
        start = time.time()
        resp = model_client.call(prompt, version=model)
        span.set_attribute("downstream.latency_ms", (time.time() - start) * 1000)
        return resp
```

*Why*: Spans let you correlate latency spikes with specific prompts or model versions, aiding root‑cause analysis.

---

**4. Edge‑case handling**  
The service validates inputs early and returns a deterministic JSON error with `error_code` `GENAI_400`.

| Edge case                | Detection logic                              | Response (`error_code`) |
|--------------------------|----------------------------------------------|--------------------------|
| Empty prompt             | `if not prompt.strip():`                     | `GENAI_400_EMPTY`        |
| Extremely long context   | `if len(prompt) > MAX_TOKENS:`               | `GENAI_400_TOO_LONG`     |
| Non‑ASCII input          | `if not prompt.isascii():`                   | `GENAI_400_NON_ASCII`    |

```json
{
  "error_code": "GENAI_400_EMPTY",
  "message": "Prompt must contain at least one non‑whitespace character."
}
```

*Failure mode*: Without these checks the model may time‑out or produce nonsensical output; returning a known error code lets callers implement retry or fallback logic consistently.

## Common Mistakes When Shipping Generative AI

- **Hard‑coding model names**  
  *Mistake*: The code imports a specific model string (e.g., `"gpt‑4"`). Deployments then break when the provider retires or upgrades the model.  
  *Mitigation*: Store the model identifier in an environment variable and gate it behind a feature‑flag service. Pin the version explicitly (`"gpt‑4.0‑20240427"`).  

  ```python
  import os
  from featureflags import is_enabled

  MODEL_ID = os.getenv("GENAI_MODEL_ID", "gpt-4.0-20240427")
  if is_enabled("use_new_model"):
      MODEL_ID = os.getenv("GENAI_NEW_MODEL_ID")
  client = GenAIClient(model=MODEL_ID)
  ```

  *Why*: Decouples code from provider changes, enabling safe roll‑backs.

- **Ignoring token‑level cost**  
  *Mistake*: Monitoring only request count hides expensive prompts that consume many tokens.  
  *Mitigation*: Emit a `cost_usd` metric per request and configure an alert when the rolling‑hour sum exceeds a budget threshold.

  ```yaml
  # monitoring.yaml
  alerts:
    - name: genai_cost_hourly
      query: sum(rate(cost_usd[5m])) by (service) > 5.0
      for: 10m
      severity: warning
  ```

  *Trade‑off*: Slightly higher telemetry volume for early cost overruns detection.

- **Relying on a single prompt template**  
  *Mistake*: One template may degrade under distribution shift.  
  *Mitigation*: Deploy an A/B testing harness that routes a configurable percentage of traffic to alternative templates and logs quality metrics (e.g., BLEU, user rating).

  **Checklist**  
  1. Define `template_A` and `template_B`.  
  2. Add a `template_variant` flag (0‑100 %).  
  3. Record `variant` alongside response and metric.  
  4. Use statistical testing to decide promotion.

  *Why*: Empirical comparison prevents silent regressions.

- **Not handling rate‑limit errors**  
  *Mistake*: A 429 response crashes the service.  
  *Mitigation*: Wrap calls with exponential backoff, jitter, and a circuit‑breaker that opens after N consecutive 429s.

  ```python
  from backoff import on_exception, expo
  from circuitbreaker import circuit

  @circuit(failure_threshold=5, recovery_timeout=60)
  @on_exception(expo, RateLimitError, max_tries=5, jitter=True)
  def generate(prompt):
      return client.complete(prompt)
  ```

  *Edge case*: If the breaker stays open, return a cached fallback or a graceful degradation message to the user.  

Applying these mitigations before launch turns common oversights into observable, controllable components.

## Testing Strategies and Production Checklist

- **Unit‑test the prompt contract validator**  
  - Verify every required field (`system`, `user`, `max_tokens`) is present.  
  - Assert that `max_tokens` does not exceed the model’s quota (e.g., 2048).  
  - Example (pytest + pydantic):

    ```python
    from myapp.prompt import PromptContract, ValidationError

    def test_prompt_contract_requires_fields():
        # missing `system` should raise
        with pytest.raises(ValidationError):
            PromptContract(user="Hello", max_tokens=100)

        # token limit breach should raise
        with pytest.raises(ValidationError):
            PromptContract(system="You are", user="Hi", max_tokens=5000)
    ```

  *Why*: catching contract violations early prevents malformed requests from reaching the LLM, saving compute cost.

- **Integration test with mocked LLM endpoint**  
  - Use `responses` or `httpx-mock` to stub `POST /v1/completions`.  
  - Simulate HTTP 429 (rate‑limit) and 500 (internal error) and assert the service falls back to a cached response or returns a graceful error payload.  

    ```python
    @mock.patch("httpx.AsyncClient.post")
    async def test_fallback_on_rate_limit(mock_post):
        mock_post.return_value = httpx.Response(429)
        resp = await client.post("/generate", json=valid_prompt)
        assert resp.status_code == 200
        assert resp.json()["fallback"] is True
    ```

  *Why*: ensures reliability when the provider throttles or experiences outages.

- **Load test with k6 (1000 concurrent users, 80th‑percentile < 500 ms)**  

    ```js
    import http from 'k6/http';
    import { check, sleep } from 'k6';
    export let options = {
      vus: 1000,
      duration: '30s',
    };

    export default function () {
      let res = http.post('https://api.myservice.com/generate', JSON.stringify({
        system: "You are a helpful assistant",
        user: "Explain k6",
        max_tokens: 150,
      }), { headers: { 'Content-Type': 'application/json' } });

      check(res, { 'status 200': (r) => r.status === 200 });
      sleep(0.1);
    }
    ```

  - After the run, verify `http_req_duration{p(80)}` < 500 ms.  
  - Trade‑off: higher VU count stresses the ingress and may expose bottlenecks in token‑budget checks; tune VUs to reflect realistic traffic.

- **Security scan checklist**  
  - Run `trivy` or `bandit` in CI to detect hard‑coded secrets in source and Docker layers.  
  - Add a log‑scrubbing step that regex‑filters patterns like `AKIA[0-9A-Z]{16}` before shipping logs to ELK.  
  - Enforce TLS termination at the ingress (e.g., NGINX Ingress with `ssl-redirect: true`). Verify with `curl -v https://api.myservice.com` that `TLSv1.3` is negotiated.  

**Production CI/CD gate**  
1. `pytest` → unit & integration ✅  
2. `k6 run load_test.js` → latency check ✅  
3. `trivy fs .` → secret scan ✅  
4. `kubectl get ingress` → TLS flag ✅  

Only when all four checks pass does the pipeline promote the build to staging. This repeatable gate guarantees functional correctness, performance under load, and baseline security before any production traffic.

## Conclusion & Next Steps

- **Prompt contract, cost‑aware inference, observability** – The contract defines a JSON schema (`prompt`, `temperature`, `max_tokens`) that every client must send; the inference wrapper enforces per‑token pricing limits and falls back to a cached response when the budget is exceeded; the observability stack (OpenTelemetry traces, Prometheus metrics, Loki logs) gives real‑time latency, error‑rate, and cost dashboards.  

- **One‑page cheat sheet** – A PDF summarizing the contract, Docker run command, and metric endpoints is available in the repo. Grab it here: [cheat‑sheet.pdf](https://github.com/yourorg/genai-mwe/blob/main/cheat-sheet.pdf).  

- **Suggested extensions** –  
  1. **Multi‑model routing** – add a selector that forwards requests to `gpt‑4o` or `llama‑2‑70B` based on token budget.  
  2. **Fine‑tuning pipeline** – automate dataset versioning, LoRA adapters, and CI‑triggered re‑deployment.  
  3. **Compliance audit checklist** – log PII detection, retain request IDs for 30 days, and encrypt logs at rest.  

- **Join the community** – Share your latency and cost numbers on the [GenAI Ops forum](https://forum.example.com). Contributions help evolve the reference implementation and keep the benchmark data fresh.
