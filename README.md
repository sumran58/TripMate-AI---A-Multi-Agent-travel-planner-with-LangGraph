# TripMate AI — Multi-Agent Travel Planner with LangGraph

TripMate AI is a multi-agent travel-planning system built on **LangGraph**. A supervisor agent screens each request, routes it to the right specialist agents (flights, hotels, weather, budget), drafts an itinerary, pauses for **human-in-the-loop approval**, and then produces a polished final plan. It's served through a **FastAPI** backend with a simple chat-style web UI, and conversations persist across turns using a **PostgreSQL**-backed LangGraph checkpointer.

## How it works

```
START
  │
  ▼
supervisor  ──(blocked)──► guardrail_blocked ──► END
  │ (allowed)
  ▼
flight_agent ──► hotel_agent ──► weather_agent ──► budget_agent   (only the agents the supervisor selects run)
  │
  ▼
itinerary_agent
  │
  ▼
human_approval  (graph pauses here via LangGraph interrupt)
  │  (resumed later with approve/reject + feedback)
  ▼
final_agent ──► END
```

1. **Supervisor Agent + Guardrail** (`supervisor_agent`) — first checks the request is actually travel-related and not harmful/off-topic (fails open if the guardrail's JSON parse fails, so a formatting hiccup never blocks a legitimate request). If allowed, it decides which specialist agents are needed and extracts trip constraints (destination, origin, duration, budget, travel style, preferences).
2. **Flight Agent** — pulls airport/airline reference data from the AviationStack MCP server and has the LLM produce route, airline, duration, fare-range, and booking guidance.
3. **Hotel Agent** — queries the hosted Tavily MCP server for hotel recommendations.
4. **Weather Agent** — calls a custom local MCP server (`custom_weather_mcp_server.py`) wrapping the OpenWeatherMap API for current conditions and a short forecast.
5. **Budget Agent** — has the LLM assess cost feasibility and savings opportunities using everything gathered so far.
6. **Itinerary Agent** — drafts a full day-by-day itinerary from all collected results and prepares it for review.
7. **Human Approval (HITL)** — the graph pauses (`interrupt()`), returning the draft itinerary to the caller. The caller resumes it later via `Command(resume=...)` with an approval flag and optional revision feedback.
8. **Final Agent** — incorporates any human feedback and produces the final, formatted response (trip summary, flights, hotels, weather, itinerary, budget, recommendations).

Only the agents the supervisor actually selects are run (the itinerary agent always runs) — a simple weather-only question skips flights/hotels/budget entirely.

## Tech stack

- **LangGraph** — `StateGraph` orchestration, conditional routing, `interrupt()` / `Command(resume=...)` for human-in-the-loop, `PostgresSaver` for durable checkpointing per `thread_id`.
- **LangChain / langchain-groq** — LLM calls via Groq (`ChatGroq`).
- **MCP (Model Context Protocol)** via `langchain-mcp-adapters` (`MultiServerMCPClient`), with three servers:
  - **Tavily MCP** (hosted, `streamable_http`) — general/hotel web search.
  - **AviationStack MCP** (`stdio`, launched with `uvx aviationstack-mcp`) — airport/airline reference data.
  - **Custom Weather MCP server** (`custom_weather_mcp_server.py`, `stdio`) — wraps OpenWeatherMap's current-weather and forecast endpoints as MCP tools.
- **FastAPI** + **Jinja2** — backend API and the chat UI (`templates/index.html`, `static/`).
- **PostgreSQL** (`psycopg`, `psycopg_pool`, `langgraph-checkpoint-postgres`) — stores graph checkpoints so a run can be paused for approval and resumed later.
- **airportsdata** / **pycountry** — resolve city/country names to IATA airport codes in `tools/flight_tool.py`.
- **Docker** — containerized deployment (`Dockerfile`).

## Project structure

```
.
├── app.py                        # FastAPI app: routes, HTML UI, health check
├── backend.py                    # LangGraph StateGraph: agents, routing, checkpointer, run/resume helpers
├── mcp_client.py                 # MultiServerMCPClient setup (Tavily, AviationStack, Weather) + tool wrappers
├── custom_weather_mcp_server.py  # Local MCP server exposing OpenWeatherMap as tools
├── tools/
│   ├── flight_tool.py            # IATA resolution + AviationStack REST search (standalone helper/CLI)
│   └── tavily_tool.py            # Direct Tavily SDK search helper (standalone)
├── templates/index.html          # Chat UI
├── static/style.css, script.js   # UI styling & frontend logic
├── mcp_client_test.py, test.py   # Manual/ad-hoc test scripts
├── requirements.txt              # pip dependencies
├── pyproject.toml / uv.lock      # uv project files
└── Dockerfile
```

## Prerequisites

- Python 3.11+
- A PostgreSQL database (e.g. a free Render Postgres instance) for LangGraph checkpointing
- [`uv`](https://docs.astral.sh/uv/) installed and on your `PATH` — the AviationStack MCP server is launched with `uvx aviationstack-mcp`, so `uvx` must be available even if you install the rest of the project with plain `pip`
- API keys: Groq, Tavily, AviationStack, OpenWeatherMap

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/sumran58/TripMate-AI---A-Multi-Agent-travel-planner-with-LangGraph.git
   cd TripMate-AI---A-Multi-Agent-travel-planner-with-LangGraph
   ```

2. **Install dependencies**

   With `uv` (the project ships a `uv.lock`):
   ```bash
   uv sync
   ```
   or with plain `pip`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   AVIATIONSTACK_API_KEY=your_aviationstack_api_key
   OPENWEATHER_API_KEY=your_openweathermap_api_key
   DATABASE_URL=postgresql://user:password@host:port/dbname

   # Optional — fallback departure airport (IATA code) when a query names
   # only a destination. Defaults to "IND" if unset.
   DEFAULT_ORIGIN_IATA=DEL
   ```
   `DATABASE_URL` must point to a reachable Postgres instance; `backend.py` will append `sslmode=require` automatically if it's missing. On first run, `PostgresSaver.setup()` creates the checkpoint tables for you.

4. **Run the app**
   ```bash
   python app.py
   # or
   uvicorn app:app --reload
   ```
   Then open `http://127.0.0.1:8000`.

### Run with Docker

```bash
docker build -t tripmate-ai .
docker run -p 8000:8000 --env-file .env tripmate-ai
```

## API

| Method | Endpoint             | Description                                                                 |
|--------|----------------------|-------------------------------------------------------------------------------|
| GET    | `/`                  | Serves the chat UI                                                            |
| POST   | `/api/travel`        | Starts (or continues) a planning run. Body: `{"message": str, "thread_id"?: str}` |
| POST   | `/api/travel/approve`| Resumes a paused run after human review. Body: `{"thread_id": str, "approved": bool, "feedback"?: str}` |
| GET    | `/health`            | Health check                                                                   |

**Example flow:**

```bash
# 1. Kick off a plan — the graph runs until it pauses for approval
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message": "Plan a complete 7 day Japan trip from Bangladesh under 2 lakhs"}'
# → { "thread_id": "...", "requires_approval": true, "itinerary": "...", ... }

# 2. Approve (or request revisions) using the returned thread_id
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "user_xxx", "approved": true}'
# → { "final_response": "..." (full formatted itinerary) }
```

Rejecting a draft requires non-empty `feedback`, which the final agent incorporates into a revised plan.

## Notes & limitations

- The Groq model is currently pinned to `openai/gpt-oss-120b` in both `backend.py` and `mcp_client.py` — swap both if you switch models.
- AviationStack's free tier returns live/status flight data, not ticket prices; `tools/flight_tool.py` labels this explicitly and suggests a pricing API (e.g. Amadeus) for real fares.
- The input guardrail *fails open*: if the guardrail LLM call or JSON parsing errors out, the request is allowed through rather than blocked, so a transient model hiccup doesn't break the app.
- `tools/flight_tool.py` and `tools/tavily_tool.py` are standalone helpers usable outside the graph (see their `if __name__ == "__main__"` blocks); the live agents in `backend.py` go through the MCP servers in `mcp_client.py` instead.

## License

MIT — see [LICENSE](LICENSE).
