# Multi-Domain Support Triage Agent

A terminal-based AI agent that triages real support tickets across three product
ecosystems — **HackerRank**, **Claude**, and **Visa** — using retrieval-augmented
generation (RAG) grounded strictly in each domain's support corpus.

For every ticket the agent decides whether it can answer safely or should escalate
to a human, classifies the request, identifies the product area, and drafts a
grounded, citation-backed response. Built for the HackerRank Orchestrate hackathon;
the full challenge spec is in [`PROBLEM_README.md`](./PROBLEM_README.md).

## Highlights

- **Agentic RAG over a graph.** Orchestrated with LangGraph as a state graph: a
  tool-using retrieval agent drives the decision, then three classification/generation
  nodes run **in parallel**.
- **Grounded by design.** The agent answers only from the provided corpus and is
  prompted not to invent policies; unsupported or high-risk tickets are escalated
  rather than guessed at.
- **Safe termination.** Custom middleware strips the agent's tools after a tool-call
  budget is reached, forcing it to commit to a structured decision instead of looping.
- **Structured, deterministic output.** Pydantic schemas + `temperature=0` keep
  results predictable and easy to evaluate against expected signals.
- **Per-domain vector stores.** Each corpus is embedded into its own Chroma collection,
  so retrieval stays scoped to the relevant ecosystem.

## Architecture

```
                         ┌───────────────────────────┐
   ticket (issue,        │   retrieve_and_decide      │
   subject, company) ───▶│  ReAct agent + 2 tools     │
                         │  • search_documentation    │  ← semantic search (Chroma)
                         │  • get_full_doc_source      │  ← full-document fetch
                         │  → decision + sources       │
                         └─────────────┬──────────────┘
                                       │  (fan-out, parallel)
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │  generate_response │  │ classify_product_area │  │ classify_request_type│
   │  grounded answer / │  │  (few-shot)           │  │  product_issue/bug/  │
   │  escalate / oos    │  │                       │  │  feature_request/... │
   └─────────┬──────────┘  └──────────┬───────────┘  └──────────┬───────────┘
             └────────────────────────┴─────────────────────────┘
                                       ▼
                    output row: status, product_area, response,
                               justification, request_type
```

**How a ticket flows through the graph:**

1. **`retrieve_and_decide`** spins up a tool-using sub-agent (LangChain `create_agent`)
   scoped to the ticket's company collection. It issues varied search queries, can pull
   full documents when an excerpt isn't enough, and emits a structured
   `RetrievalDecision` of `reply` / `escalate` / `out_of_scope` with a one-line rationale
   and the source paths that actually resolve the issue. A tool-call-limit middleware
   guarantees the loop ends with a decision.
2. The graph **fans out in parallel** to three independent nodes:
   - **`generate_response`** — writes a concise, corpus-grounded answer (or an escalation /
     out-of-scope message).
   - **`classify_product_area`** — maps the issue to a product area, guided by few-shot
     examples spanning all three domains.
   - **`classify_request_type`** — labels the ticket `product_issue`, `feature_request`,
     `bug`, or `invalid`.
3. Results merge into a single output row written to `support_tickets/output.csv`.

A deliberate design choice: only the documents the agent deems *necessary to resolve*
the ticket are carried forward as `relevant_documents`, separate from everything it
merely looked at. This keeps the downstream response and classifications tightly grounded.

## Tech stack

- **Python 3.11+**
- **LangGraph** — state-graph orchestration with parallel node execution
- **LangChain** — agent loop, tools, middleware, structured output, prompt templating
- **Chroma** — local persistent vector stores (one per domain)
- **OpenAI** — chat model + embeddings (configurable in `code/config.py`)
- **Pydantic** — structured decision/classification schemas
- **uv** — dependency management

## Setup

```bash
cd code/
cp .env.example .env          # then add a valid OPENAI_API_KEY
uv sync                       # install deps (see https://docs.astral.sh/uv/)
source .venv/bin/activate

python ingestion.py           # build the per-domain Chroma vector stores
python main.py                # run the agent over support_tickets.csv → output.csv
```

Secrets are read from environment variables only (`OPENAI_API_KEY`); nothing is
hardcoded, and `.env` is gitignored.

`main.py` exposes a few run knobs at the top: `SAMPLE` (run the labeled sample set),
`RUN_PARALLEL` + `MAX_CONCURRENCY` (process tickets concurrently with an asyncio
semaphore).

## Project structure

```
code/
├── main.py                 # entry point: read tickets → run graph → write output
├── config.py               # paths and model configuration
├── ingestion.py            # corpus loading, chunking, embedding into Chroma
├── program_io.py           # CSV read/write with a fixed output schema
└── graph/
    ├── graph.py            # LangGraph wiring (entry node + parallel fan-out)
    ├── state.py            # typed GraphState (inputs, internal state, outputs)
    ├── utils.py            # document formatting + prompt helpers
    └── nodes/
        ├── retrieve_and_decide.py   # tool-using agent + decision + middleware
        ├── generate_response.py     # grounded response generation
        ├── classify_product_area.py # few-shot product-area classification
        └── classify_request_type.py # request-type classification
data/                       # support corpora: claude/, hackerrank/, visa/
support_tickets/            # input CSVs + generated output.csv
```

## Possible improvements

- Few-shot examples for request-type classification (currently relies on instructions only).
- Confidence scoring on the retrieval decision to tune the reply/escalate threshold.
- Caching embeddings and retrieval results to cut repeat-run cost and latency.
- An automated eval harness scoring `output.csv` against the labeled sample set.
