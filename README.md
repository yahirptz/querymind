# QueryMind — AI-Powered SQL Assistant


---

## Why QueryMind Instead of Just Asking an LLM?

You might be thinking: *"Can't I just paste my data into Claude or ChatGPT and ask questions?"*

Yes — but QueryMind solves real problems that raw LLM chat cannot:

**Your data stays private.** Pasting sensitive business data, customer records, or financial spreadsheets into a public AI chat means that data leaves your machine. QueryMind runs entirely locally on your computer inside Docker. Your data never leaves your infrastructure.

**It works on data too large for a chat window.** LLMs have context limits. A 50,000 row CSV or a database with millions of records cannot be pasted into a chat. QueryMind queries the actual database and only sends the schema — not the data — to Claude, so it works at any scale.

**It executes real queries, not guesses.** When you ask Claude in a chat "what was my best sales month?", it estimates based on what you pasted. QueryMind generates actual SQL, runs it against your real database, and returns exact results from your actual data.

**It remembers your schema automatically.** Every time you upload a new CSV or connect a database, QueryMind embeds the schema into its vector store. You never have to re-explain your data structure — just ask questions.

**It keeps a full query history.** Every question, the SQL it generated, and the result count are saved. You can review, audit, and learn from past queries.

---

## What It Does

1. Connect to a PostgreSQL database or upload any CSV file
2. Ask a question in plain English
3. QueryMind retrieves relevant schema context via semantic search (RAG)
4. Sends schema + question to Claude to generate a PostgreSQL SELECT query
5. Executes the query safely in a read-only transaction
6. Returns the raw results and a plain-English summary
7. Saves every query to history for review

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
│    ├─1─► pgvector semantic search (primary RAG path)           │
│    │      sentence-transformers all-MiniLM-L6-v2               │
│    │                                                            │
│    ├─2─► information_schema fallback (if pgvector misses any)  │
│    │      guarantees Claude always sees complete schema         │
│    │                                                            │
│    ├─3─► claude_client ──────► Anthropic API (SQL generation)  │
│    │                                                            │
│    ├─4─► db.py ──────────────► PostgreSQL (read-only exec)     │
│    │                                                            │
│    ├─5─► claude_client ──────► Anthropic API (summarization)   │
│    │                                                            │
│    └─6─► models.py ──────────► PostgreSQL (query history)      │
│                                                                 │
│  POST /api/upload-csv  — upload CSV, auto-embed schema         │
│  GET  /api/schema      — schema tree for sidebar               │
│  GET  /api/history     — last 20 queries                       │
│  GET  /health          — liveness check                        │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL 15 + pgvector extension                 │
│                                                                 │
│  chinook tables        — sample music store data               │
│  user csv tables       — any CSV you upload                    │
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
| AI — SQL + Summary | Anthropic Claude Sonnet |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| ORM | SQLAlchemy 2.0 |
| Frontend | Vanilla JS, single HTML file |
| Containerization | Docker, Docker Compose |
| Sample Data | Chinook music store database |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- An [Anthropic API key](https://console.anthropic.com/)

> **Note:** First build takes 10-15 minutes as Docker downloads all dependencies and the embedding model. Every start after that takes under 30 seconds.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yahirptz/querymind
cd querymind

# 2. Create your environment file
cp .env.example .env

# 3. Edit .env — add your Anthropic API key and set a Postgres password
#    ANTHROPIC_API_KEY=sk-ant-...
#    POSTGRES_PASSWORD=your_secure_password

# 4. Start all services
docker compose up --build

# 5. Seed the sample Chinook database (first time only)
docker compose exec app python scripts/load_chinook.py

# 6. Open the app
http://localhost:5000
```

Schema embeddings are generated automatically on first startup. No manual step needed.

### Development mode (with pgAdmin)

```bash
docker compose --profile dev up
# pgAdmin available at http://localhost:5050
# Login: admin@admin.com / admin
```

---

## Uploading Your Own Data

QueryMind accepts any CSV file — exported from Excel, Google Sheets, or any tool.

1. Click the **Data** tab in the top navigation
2. Drag and drop your CSV file or click to browse
3. Give the table a name
4. Click **Upload**

The schema is automatically embedded into pgvector so you can start asking questions immediately. Tables appear in the schema sidebar and can be deleted at any time.

**Limits:** 10 MB max file size, 50,000 rows max.

---

## Usage Examples

### Chinook sample database

| Question | What it demonstrates |
|---|---|
| `What are the top 5 selling artists?` | Aggregation + JOIN across multiple tables |
| `How many customers are from the USA?` | Simple filter + COUNT |
| `What is the total revenue by country?` | GROUP BY + SUM |
| `Which albums have more than 15 tracks?` | HAVING clause |
| `Who are the top 3 customers by total spending?` | Subquery + ORDER BY LIMIT |
| `What genre has the most tracks?` | JOIN + GROUP BY + ORDER BY |

### With your own CSV

| Question |
|---|
| `What was my best month for sales?` |
| `Which product category has the highest average price?` |
| `How many records have missing values in the email column?` |
| `What is the average house price by number of rooms?` |

---

## How the RAG Pipeline Works

1. **User submits a question** via the chat UI
2. **Question embedding** — encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Semantic schema retrieval** — pgvector cosine similarity search returns the most relevant table descriptions
4. **Fallback layer** — any tables missed by pgvector are fetched directly from `information_schema`, guaranteeing complete context
5. **SQL generation** — Claude receives the question + full schema context and responds with a safe PostgreSQL SELECT query
6. **Safety validation** — SQL is checked: must start with SELECT, no DDL/DML keywords, no comments, under 2000 chars
7. **Read-only execution** — query runs in a `SET TRANSACTION READ ONLY` transaction
8. **Result summarization** — Claude produces a plain-English summary of the results
9. **History saved** — question, SQL, and row count written to `query_history`
10. **Response returned** — UI displays summary, collapsible SQL, and a formatted results table

---

## Project Structure

```
.
├── docker-compose.yml        # Service definitions (app, db, pgadmin)
├── Dockerfile                # Python 3.11-slim + gunicorn
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── app/
│   ├── main.py               # Flask factory + all routes
│   ├── db.py                 # SQLAlchemy engine, query execution, schema metadata
│   ├── schema_parser.py      # Schema → natural language chunks for embedding
│   ├── embeddings.py         # sentence-transformers + pgvector store/retrieval
│   ├── claude_client.py      # Anthropic SDK wrapper (SQL gen + summarization)
│   ├── rag_pipeline.py       # Orchestration: RAG → validate → execute → summarize
│   └── models.py             # QueryHistory ORM model
├── static/
│   └── index.html            # Dark-theme chat UI (HTML + CSS + JS, single file)
├── scripts/
│   ├── load_chinook.py       # Seeds Chinook data into PostgreSQL
│   └── embed_schema.py       # Embeds schema into pgvector
└── tests/
    ├── test_db.py            # DB layer + SQL validation tests
    └── test_rag_pipeline.py  # Pipeline orchestration + schema parser tests
```

---

## Running Tests

```bash
# Inside the container
docker compose exec app pytest tests/ -v
```

---

## Security

- All SQL execution uses `SET TRANSACTION READ ONLY`
- Generated SQL validated against a forbidden-keyword blocklist before execution
- API keys loaded exclusively from `.env` — never hardcoded
- Errors returned to the frontend never include raw stack traces
- Non-root Docker user (`appuser`) runs the application
- CSV uploads sanitized: table names and column names are cleaned before use
- Protected table list prevents overwriting or deleting core database tables

---

## Future Improvements

- [ ] Authentication + per-user query history
- [ ] Query result export (CSV / JSON download)
- [ ] Streaming responses for long-running queries
- [ ] Query result caching with Redis
- [ ] Support for MySQL, SQLite, and Snowflake connections
- [ ] Natural language schema exploration ("Tell me about the invoices table")
- [ ] Fine-tuned embedding model for SQL schema domains

---

## License

MIT License — Copyright (c) 2026 Yahir Perez Tecalco