# AI-Powered SQL Query Assistant

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://postgresql.org)
[![pgvector](https://img.shields.io/badge/pgvector-0.3-336791)](https://github.com/pgvector/pgvector)
[![Anthropic](https://img.shields.io/badge/Claude-Sonnet-D4A017)](https://anthropic.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)

A production-quality **RAG-based web application** that lets users ask natural language questions about a relational database and receive accurate SQL queries + plain-English answers — powered by Claude (Anthropic) and semantic search via pgvector.


---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                           │
│                  (Chat UI — static/index.html)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP (fetch API)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Backend (app/)                         │
│                                                                 │
│  POST /api/query                                                │
│    │                                                            │
│    ├─1─► schema_parser  ◄──── PostgreSQL (schema metadata)     │
│    │                                                            │
│    ├─2─► embeddings.py  ◄──── pgvector (cosine similarity)     │
│    │      sentence-transformers (all-MiniLM-L6-v2)              │
│    │                                                            │
│    ├─3─► claude_client  ──────► Anthropic API (SQL generation) │
│    │                                                            │
│    ├─4─► db.py          ──────► PostgreSQL (read-only exec)    │
│    │                                                            │
│    ├─5─► claude_client  ──────► Anthropic API (summarization)  │
│    │                                                            │
│    └─6─► models.py      ──────► PostgreSQL (query history)     │
│                                                                 │
│  GET /api/schema    — schema tree for sidebar                   │
│  GET /api/history   — last 20 queries                           │
│  GET /health        — liveness check                            │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL 15 + pgvector extension                 │
│                                                                 │
│  chinook tables        — music store sample data               │
│  schema_embeddings     — vector store for RAG                  │
│  query_history         — audit log of all queries              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3, Gunicorn |
| Database | PostgreSQL 15, pgvector |
| AI (SQL + Summary) | Anthropic Claude Sonnet |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| ORM | SQLAlchemy 2.0 |
| Frontend | Vanilla JS, single HTML file |
| Containerization | Docker, Docker Compose |
| Sample Data | Chinook (music store) |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd sql-query-assistant

# 2. Create your environment file
cp .env.example .env

# 3. Edit .env and add your Anthropic API key
#    ANTHROPIC_API_KEY=sk-ant-...
#    Also set a secure POSTGRES_PASSWORD

# 4. Start all services
docker compose up --build

# 5. Seed the Chinook sample database (first time only)
docker compose exec app python scripts/load_chinook.py

# 6. Open the app
open http://localhost:5000
```

> Schema embeddings are generated automatically on first startup. No manual step needed.

### Development mode (with pgAdmin)

```bash
docker compose --profile dev up
# pgAdmin available at http://localhost:5050
# Login: admin@admin.com / admin
```

---

## Usage Examples

Once the app is running, try these questions in the chat interface:

| Question | What it demonstrates |
|---|---|
| `What are the top 5 selling artists?` | Aggregation + JOIN across multiple tables |
| `How many customers are from the USA?` | Simple filter + COUNT |
| `What is the total revenue by country?` | GROUP BY + SUM |
| `Which albums have more than 15 tracks?` | HAVING clause |
| `Who are the top 3 customers by total spending?` | Subquery + ORDER BY LIMIT |
| `What genre has the most tracks?` | JOIN + GROUP BY + ORDER BY |

---

## How the RAG Pipeline Works

1. **User submits a question** via the chat UI
2. **Question embedding** — the question is encoded into a 384-dim vector using `all-MiniLM-L6-v2`
3. **Semantic schema retrieval** — cosine similarity search over `schema_embeddings` in pgvector returns the top-5 most relevant table descriptions
4. **SQL generation** — Claude receives the question + schema context and responds with a safe PostgreSQL SELECT query
5. **Safety validation** — the SQL is checked: must start with SELECT, no DDL/DML keywords, no comments, under 2000 chars
6. **Read-only execution** — query runs in a `SET TRANSACTION READ ONLY` transaction
7. **Result summarization** — Claude produces a plain-English summary of the results
8. **History saved** — question, SQL, and row count are written to `query_history`
9. **Response returned** — UI displays summary, collapsible SQL, and a formatted results table

---

## Project Structure

```
.
├── docker-compose.yml        # Service definitions (app, db, pgadmin)
├── Dockerfile                # Python 3.11-slim + gunicorn
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # Flask factory + routes
│   ├── db.py                 # SQLAlchemy engine, query execution, schema metadata
│   ├── schema_parser.py      # Schema → natural language chunks
│   ├── embeddings.py         # sentence-transformers + pgvector store/retrieval
│   ├── claude_client.py      # Anthropic SDK wrapper (SQL gen + summarization)
│   ├── rag_pipeline.py       # Orchestration: validate → RAG → execute → summarize
│   └── models.py             # QueryHistory ORM model
├── static/
│   └── index.html            # Dark-theme chat UI (HTML + CSS + JS, single file)
├── scripts/
│   ├── load_chinook.py       # Seeds Chinook data into PostgreSQL
│   └── embed_schema.py       # Embeds schema into pgvector (auto-run on startup)
└── tests/
    ├── test_db.py            # DB layer + SQL validation tests
    └── test_rag_pipeline.py  # Pipeline orchestration + schema parser tests
```

---

## Running Tests

```bash
# Inside the container
docker compose exec app pytest tests/ -v

# Or locally (requires a running DB)
pip install -r requirements.txt
pytest tests/ -v
```

---

## Security

- All SQL execution uses `SET TRANSACTION READ ONLY`
- Generated SQL is validated against a forbidden-keyword blocklist before execution
- API keys are loaded exclusively from `.env` — never hardcoded
- Errors returned to the frontend never include raw stack traces
- Non-root Docker user (`appuser`) runs the application

---

## Future Improvements

- [ ] Multi-database support (MySQL, SQLite, Snowflake)
- [ ] Authentication + per-user query history
- [ ] Query result export (CSV / JSON download)
- [ ] Schema change detection + automatic re-embedding
- [ ] Streaming responses for long-running queries
- [ ] Query result caching with Redis
- [ ] Fine-tuned embedding model for SQL schema domains
- [ ] Natural language schema exploration ("Tell me about the invoices table")

---

## License

MIT License — see [LICENSE](LICENSE) for details.
