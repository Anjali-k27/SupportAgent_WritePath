# SupportAgent_WritePath# Enterprise AI Support Platform
### Session 10 of 12 — System API Integration: Write Access

A LangGraph-powered multi-agent customer support system. Tickets are classified, routed through parallel specialist agents, and critical findings are automatically escalated to GitHub issues.

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/waseemkhan606/phase3-session9
cd phase3-session9
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the spaCy language model** (required by Presidio)
```bash
python -m spacy download en_core_web_lg
```

**5. Set your API key**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-gemini-api-key-here
```

Or export it directly:
```bash
export GOOGLE_API_KEY=your-gemini-api-key-here
```

Get a key at: https://aistudio.google.com/app/apikey

---

## Running

**Start the web server**
```bash
python api.py
```
Then open http://localhost:8000 in your browser.

**Run the CLI test suite**
```bash
python support_agent.py
```

---

## GitHub Integration (optional)

To create real GitHub issues instead of mock ones, set two additional environment variables:

```bash
export GITHUB_TOKEN=ghp_your_personal_access_token
export GITHUB_REPO=your-org/your-repo
```

Without these, the system runs in **mock mode** — all GitHub issue logic executes but no real issues are created. The mock URL is shown in the UI exactly as a real URL would be.

To generate a GitHub token: Settings → Developer settings → Personal access tokens → Generate new token. Required scope: `repo`.

---

## Architecture

```
Ticket
  └── ingress_node        (PII masking, injection detection)
        └── classify_node (technical / billing / fraud / general)
              ├── dispatcher ──────────────────── [technical / fraud / multi-issue]
              │     ├── tech_analysis_agent
              │     │     └── github_tool_node   (high/critical → GitHub issue)
              │     │           └── synthesizer  (unified response)
              │     ├── billing_analysis_agent ──┘
              │     └── fraud_analysis_agent ────┘
              └── supervisor ─────────────────── [billing / general]
                    ├── tech_support subgraph
                    ├── fraud_handler
                    └── general_handler
```

**Key components:**

| File | Purpose |
|------|---------|
| `support_agent.py` | Agent graph, all nodes, tools, state schema |
| `api.py` | FastAPI backend — REST + SSE streaming endpoints |
| `index.html` | Single-page frontend — live execution inspector |
| `requirements.txt` | Python dependencies |

---

## How it works

1. Every ticket passes through `ingress_node` — PII is masked by Presidio, injection patterns are blocked.
2. `classify_node` routes to one of four categories: technical, billing, fraud, general.
3. **Technical tickets** go to the dispatcher, which fans out to parallel specialist agents simultaneously using LangGraph's Send API.
4. `tech_analysis_agent` searches the knowledge base. If it finds a **critical or high severity** issue, it populates `github_draft` in state.
5. `github_tool_node` reads the draft, generates an idempotency key (SHA-256 of `thread_id + title + body`), and calls the GitHub API. On retry, the same key is produced — no duplicate issues.
6. `synthesizer_node` merges all findings and produces a single coherent customer response, including the GitHub issue URL if one was created.
7. **Billing/general tickets** go to the supervisor path — an LLM-powered hub-and-spoke orchestrator that delegates to workers and decides when the ticket is resolved.

---

## Verification

Run the automated test suite through the UI by clicking **Run Verification Test** at the bottom of the page, or via the API:

```bash
curl -X POST http://localhost:8000/api/verify | python -m json.tool
```

Expected output: **5/5 checks passed**

| Check | What it proves |
|-------|---------------|
| Idempotency key is deterministic and unique | Same inputs always produce the same key; different inputs always produce different keys |
| Critical ticket → github_draft populated | tech_analysis_agent correctly identifies critical findings |
| github_issue_url set after github_tool_node | The write path executes end-to-end |
| Second call same thread → no duplicate issue | Idempotency protection works across retries |
| GitHub failure → structured error, no crash | Graceful degradation keeps the system functional |

---

## Session Progress

| Session | Topic | Status |
|---------|-------|--------|
| 1 | The Blueprint | ✅ |
| 2 | Tool Binding | ✅ |
| 3 | The ReAct Architecture | ✅ |
| 4 | Persistence & Threading | ✅ |
| 5 | Context Management | ✅ |
| 6 | Guardrails & Bounding | ✅ |
| 7 | Multi-Agent Topologies | ✅ |
| 8 | The Supervisor Orchestrator | ✅ |
| 9 | Shared Scratchpads & Consensus | ✅ |
| 10 | System API Integration: Write Access | 🟢 **current** |
| 11 | Interruptions & Breakpoints | 🔒 |
| 12 | The Auditor | 🔒 |
