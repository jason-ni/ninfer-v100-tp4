# Request persistence (remote request log)

Status: implemented and verified. `ninfer-serve` ships a copy of each parsed OpenAI
`/v1/chat/completions` request body to a remote **request-persistence server** over a small
length-prefixed TCP binary protocol; a Python asyncio receiver extracts the JSON body and stores
it in SQLite.

## Overview

When the serve HTTP handler accepts a well-formed chat request for the served model, it copies the
raw request body out of the hot path and hands it to an in-memory queue. A dedicated sender thread
drains that queue to a remote TCP endpoint.

Persistence must never block or fail request serving: enqueue is non-blocking (the record is
dropped, and counted, when the queue is full) and all network I/O happens on one background thread.
The remote receiver is a long-lived service that outlives any single serve process.

The transport is split into two owning components:

- C++ (`src/serve/request_persist.{h,cpp}`): `RequestPersistor` — a bounded thread-safe queue, a
  dedicated sender thread, and a raw POSIX TCP client with connect timeout, `TCP_NODELAY`, and
  reconnect backoff — plus the pure, transport-agnostic wire codec
  (`encode_persist_frame` / `decode_persist_frame`).
- Python (`tools/request_persist/request_persist_server.py`): an asyncio TCP server with a frame
  reader and a single-writer, content-addressed SQLite path (per-record commit, graceful shutdown),
  plus a stdlib `http.server` REST admin interface on a separate port for querying and deleting
  stored records, and a self-contained single-page web UI served at `GET /` on that same port.

The length prefix and the envelope are handled separately so the codec stays unit-testable on both
sides. A golden-byte vector pins the exact encoding and is asserted identically by the C++ and
Python tests.

## Wire protocol (shared contract)

Big-endian integers. One record = one frame on the stream.

```
Frame      = u32be(envelope_len) || envelope
Envelope   = version(1) flags(1)
             request_id(u64be)
             created_unix_ms(i64be)
             protocol_len(u32be) protocol[protocol_len]
             model_len(u32be)    model[model_len]
             body_len(u32be)     body[body_len]
```

- `version` = 1 (`kPersistProtocolVersion`); bump on any envelope change, the receiver rejects
  unknown versions.
- `flags` = 0 (reserved).
- `request_id` = the serve-side per-process sequence id.
- `created_unix_ms` = wall-clock ms at enqueue time.
- `protocol` = e.g. `"openai_chat_completions"`.
- `model` = the served model id.
- `body` = the raw JSON request body exactly as received (unmodified).

## Storage schema (SQLite)

Bodies are stored **content-addressed**, not verbatim. Agent traffic resends the whole chat history
(and the full `tools` schema) on every turn, so the same bytes recur across requests and across
branches. A body is split into exact byte spans; each span is interned once in `chunks`; a request
row references them, and a body is reconstructed byte-for-byte on read.

```sql
CREATE TABLE chunks (               -- content-addressed byte spans (sha256)
  id     INTEGER PRIMARY KEY AUTOINCREMENT,
  hash   BLOB    NOT NULL UNIQUE,   -- sha256(raw), the content address
  raw    BLOB    NOT NULL,          -- exact byte span (framing or one message)
  nbytes INTEGER NOT NULL,
  refs   INTEGER NOT NULL DEFAULT 0 -- live request references, for GC
);
CREATE TABLE requests (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id      INTEGER NOT NULL,   -- serve-side per-process seq
  created_unix_ms INTEGER NOT NULL,   -- serve enqueue time
  protocol        TEXT    NOT NULL,
  model           TEXT    NOT NULL,
  message_count   INTEGER,            -- extracted len(body["messages"]) if present
  stream          INTEGER,            -- extracted body["stream"] (0/1) if present
  received_at     TEXT    NOT NULL,   -- ISO8601 UTC receiver time
  head_id         INTEGER NOT NULL,   -- chunks: bytes up to and incl. the messages '['
  tail_id         INTEGER NOT NULL,   -- chunks: bytes from the last element's end
  msg_ids         BLOB    NOT NULL,   -- varint-packed ordered message chunk ids
  body_len        INTEGER NOT NULL    -- reconstructed body size in bytes
);
CREATE INDEX idx_requests_model   ON requests(model);
CREATE INDEX idx_requests_created ON requests(created_unix_ms);
```

A body `{"model":...,"tools":[...],"messages":[m0,m1,...]}` splits into `head` (everything through
the `[`, which carries the repeated `tools` block), one span per message (each carrying its leading
separator, so appending a message never rewrites an earlier span), and `tail` (whitespace + `]` +
the rest). Interning is by content hash, so a shared system prompt, `tools` schema, or earlier
message is stored once no matter how many requests (or retries, or sessions) reference it — branches
and cross-session overlap dedup with no parent-pointer bookkeeping. A body without a top-level
`messages` array is stored whole as `head`.

Deletion decrements each referenced chunk's `refs` and drops chunks that reach zero, so removing one
request never deletes history still referenced by another.

## Deduplication

For the agent workload the store is proportional to *unique* content, not to request count. In a
representative `qwen3.8-27b` log, 309 requests (~99 MB of bodies) collapse to ~1.9 MB of chunks
(651 unique spans, ~43× smaller): the `tools` schema is identical across nearly all requests, and
almost every message reappears on the next turn. `GET /stats` reports `unique_chunks` and
`chunk_bytes`; compare against `body_chars.total` for the ratio. Because content is shared,
`DELETE ...&vacuum` and `POST /vacuum` also reclaim any chunks whose reference count has dropped to
zero.

A legacy database written by the previous verbatim (`body_json`) schema is rebuilt into chunks and
`VACUUM`-compacted automatically on first open (one-time); legacy bodies that were truncated by the
removed ingest cap are retained as-is (their lost tail cannot be recovered).

## REST admin API

The receiver also serves a small JSON REST interface (stdlib `http.server`, threaded) on a separate
port — default `127.0.0.1:9101` (`--http-host` / `--http-port`; `--http-port 0` binds an ephemeral
port, `--no-http` disables it). Each request opens its own short-lived SQLite connection; the connect
timeout doubles as the busy timeout that absorbs the brief lock wait when a read/delete contends with
the single serial insert writer. The DB keeps its existing rollback journal mode, so `VACUUM`
reclaims space by rewriting the main file — which is exactly what the `*_bytes` fields report.

Responses are `application/json`. Errors are `{"error": "<msg>"}` with `400` (bad parameters),
`404` (unknown route or missing id), or `500` (database error).

| method & path | purpose |
|---|---|
| `GET /health` | liveness; returns db path, ports, protocol version |
| `GET /stats` | row count, db file size, created/received ranges (+ inferred unit), per-model/protocol counts, body-size stats |
| `GET /requests` | list/search with filters, pagination, ordering |
| `GET /requests/{id}` | fetch one record by autoincrement `id` |
| `DELETE /requests` | delete by criteria; optional `dry_run` and `vacuum` |
| `POST /vacuum` | reclaim disk space with `VACUUM` without deleting |

Filters for `GET /requests` and `DELETE /requests` (AND-combined, single value each): `model`,
`protocol`, `stream` (`0`/`1`), `min_messages`/`max_messages`, `min_body_kb`/`max_body_kb`, `id`,
`ids` (comma-separated), and time bounds `before`/`after` plus `older_than_days`/`newer_than_days`.
`GET /requests` additionally takes `limit` (default 50, max 1000), `offset`, `order`
(`asc`/`desc`, default newest-first) and `include_body` (default `true`; set `false` to browse many
rows without shipping the large bodies). Every row carries `body_bytes` (its stored size) so
oversized records can be targeted directly. `DELETE` with no criterion and no `all=true` returns `400`.

### Web UI (SPA)

The receiver also serves a small single-page app at `GET /` (and `/index.html`) on the **same** admin
port — e.g. `http://127.0.0.1:9101/`. It is a self-contained `tools/request_persist/request_persist_ui.html`
(vanilla HTML/CSS/JS, no CDN, no build step, no extra process) read once at startup; if the file is absent
`GET /` returns `503` while TCP ingest and the JSON API keep working. The page calls the exact JSON endpoints
above (`/health`, `/stats`, `/requests`, `DELETE /requests`, `/vacuum`), so it adds no second data path.

It provides:

- a live stats strip (rows, DB file size, created range + stored unit, per-model/protocol counts, unique chunks + stored bytes);
- the same AND-combined filters as the REST API (model, protocol, stream, message range, body-size range,
  `id`/`ids`, `before`/`after`/age) with pagination and newest/oldest ordering;
- a click-to-open record viewer that fetches one row and pretty-prints its full body, with Copy/Download;
- **delete-with-preview** (always runs a dry-run and shows the would-delete count before you confirm the real
  delete + `VACUUM`) and a standalone vacuum.

Listing uses `include_body=false` for speed; bodies are fetched only when a record is opened. `--no-http`
disables both the JSON API and the web UI.

### Time bounds and the units trap

`created_unix_ms` is compared against its **stored unit**, which the API infers from the column's
maximum value: values below `1e12` are treated as whole seconds, at/above it as milliseconds. This
matters because the C++ producer currently writes **seconds** into the field despite the
milliseconds-oriented name. Age filters (`older_than_days` etc.) compute their cutoff in the inferred
unit, so they behave correctly either way. `before`/`after` accept either a bare integer (assumed
already in the stored unit) or an ISO-8601 date/datetime (naive ⇒ UTC; `Z`/offset honored).

> Follow-up (not part of this API): the wire field `created_unix_ms` and its `openai_schema.cpp::
> unix_time_now()` source produce whole **seconds**, not milliseconds. Aligning the producer to true
> milliseconds is a change to the shared wire contract (golden byte vector + both C++/Python test
> suites) and is tracked separately; the REST layer stays correct today via unit inference.

### Cleanup workflow

Because bodies are deduplicated by content, the DB grows with unique content rather than with
request count. Deletion still leaves free pages behind, so pair deletes with `vacuum` (which also
GCs unreferenced chunks):

```bash
BASE=http://127.0.0.1:9101
curl -s "$BASE/stats"                                        # rows, file size, ranges, per-model
curl -s "$BASE/requests?older_than_days=7&limit=5&include_body=false"   # preview recent matches
curl -s -X DELETE "$BASE/requests?older_than_days=7&dry_run"           # would-delete count, no change
curl -s -X DELETE "$BASE/requests?older_than_days=30&vacuum"           # delete + reclaim in one step
curl -s -X DELETE "$BASE/requests?min_body_kb=5120&vacuum"             # drop only the oversized rows
curl -s -X POST "$BASE/vacuum"                                        # standalone space reclamation
```

`DELETE ...&vacuum` returns `{deleted, dry_run, vacuumed, bytes_before, bytes_after, bytes_freed}`.
A long `VACUUM` on a large file briefly holds the write lock, so any concurrent ingest pauses until
it finishes (records are dropped only if the bounded queue fills); that is expected for a manual
cleanup action.

## CLI surface

`ninfer-serve`:

- `--persist-endpoint HOST:PORT` — enable persistence and set the remote target (empty/unset =>
  disabled).
- `--persist-queue-capacity N` — bounded in-memory queue size (default 4096; positive).
- `--persist-reconnect-ms N` — backoff between reconnect attempts (default 1000; positive).

`run-tp4.sh`:

- `--remote-req-log[=HOST:PORT]` — ensure the Python receiver is running (spawning it detached when
  its port is not already listening), store the DB under `/media/optane/reqlogs`, and forward
  `--persist-endpoint` to `ninfer-serve`. The TCP ingest listener is bound to the endpoint `HOST`
  (loopback by default); the REST admin + web UI bind to `0.0.0.0` by default so they are reachable
  from other hosts, on port `9101`. Overridable with `NINFER_REMOTE_REQ_LOG_ENDPOINT`,
  `NINFER_REMOTE_REQ_LOG_HTTP_HOST`, `NINFER_REMOTE_REQ_LOG_HTTP_PORT`,
  `NINFER_REMOTE_REQ_LOG_DIR`, `NINFER_REMOTE_REQ_LOG_DB`,
  and `NINFER_REMOTE_REQ_LOG_PYTHON`.

`receiver (tools/request_persist/request_persist_server.py)`:

- `--http-host ADDR` / `--http-port N` — REST admin + web UI bind; the flag defaults to `127.0.0.1:9101`
  (`0` = ephemeral port), while `run-tp4.sh` passes `0.0.0.0` so the API and UI are reachable off-box.
  The TCP ingest listener is bound separately via `--host` and is not exposed by this setting.
- `--no-http` — disable the REST admin interface (TCP ingest still runs).

## Shutdown

The receiver is a long-lived shared service: it is spawned detached and left running across serve
restarts. On the serve side the persistor is flushed by `~RequestPersistor` (destructor), never from
the signal-driven `HttpServer::stop()` — a signal handler must not join a thread. httplib already
joins all HTTP handler threads before `listen()` returns, and `HttpServer` member order destroys the
HTTP server before the persistor, so the destructor flush is race-free.

## How to run

```bash
# Terminal A: receiver
python3 tools/request_persist/request_persist_server.py \
    --host 127.0.0.1 --port 9100 --db /media/optane/reqlogs/request_persist.sqlite

# Terminal B: serve with persistence enabled
build-v100/apps/ninfer-serve /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer \
    --device <free-gpu> --persist-endpoint 127.0.0.1:9100

# Terminal C: fire a request
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"<public-model-id>","messages":[{"role":"user","content":"hi"}]}'

# Verify
sqlite3 /media/optane/reqlogs/request_persist.sqlite \
    'SELECT request_id,model,message_count FROM requests;'
```

On the V100X4 launcher the whole receiver setup is one flag:

```bash
./run-tp4.sh --remote-req-log
```

Once the receiver is up, open the web UI at `http://127.0.0.1:9101/` in a browser to browse, filter,
inspect, and clean up stored requests without `curl`. The admin port binds `0.0.0.0` (via `run-tp4.sh`),
so it is also reachable from other hosts at the machine's LAN address — e.g. `http://<host-ip>:9101/`
(see `hostname -I`). The admin port is unauthenticated and supports delete/vacuum, so keep it to a
trusted network.

## Verification

- `ninfer_request_persist_test`: codec golden bytes, round-trip, and reject cases — ok.
- `ninfer_serve_options_test`: new-flag defaults, parsing, invalid values, and usage — ok.
- Python `pytest tests/test_request_persist.py`: 33 passed — golden bytes matching the C++ vector,
  decode/round-trip, frame reader, field extraction, byte-exact body split (including irregular
  whitespace, empty/absent `messages`, and multibyte text), append-stability of message spans,
  varint id round-trip, content-addressed dedup (growing history collapses to a handful of chunks),
  lossless reconstruction through insert/get/query, reference-count GC on delete, legacy
  `body_json`-schema migration, a real TCP loopback end-to-end into SQLite, REST unit coverage
  (filter builder, ISO/epoch/unit parsing, flag semantics), web-UI serving (`GET /` and `/index.html`
  return `text/html` with the page marker over a live ephemeral-port server), and a full TCP+REST
  end-to-end (insert large bodies → stats/list/get → delete by age, dry-run, all+vacuum) over real
  sockets.
- GPU end-to-end (real engine, qwen3.8-27b, TP2): two `/v1/chat/completions` requests (1-message and
  3-message history) completed; both rows landed in SQLite with the exact raw bodies
  (`message_count` 1 then 3); SIGTERM flushed the persistor and the receiver logged
  `connection ... closed (2 records)`.
