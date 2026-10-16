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
  reader and a single-writer SQLite path (per-record commit, graceful shutdown).

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

```sql
CREATE TABLE requests (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id      INTEGER NOT NULL,   -- serve-side per-process seq
  created_unix_ms INTEGER NOT NULL,   -- serve enqueue time
  protocol        TEXT    NOT NULL,
  model           TEXT    NOT NULL,
  body_json       TEXT    NOT NULL,   -- raw JSON body (source of truth)
  message_count   INTEGER,            -- extracted len(body["messages"]) if present
  stream          INTEGER,            -- extracted body["stream"] (0/1) if present
  received_at     TEXT    NOT NULL    -- ISO8601 UTC receiver time
);
CREATE INDEX idx_requests_model   ON requests(model);
CREATE INDEX idx_requests_created ON requests(created_unix_ms);
```

The full `body_json` is retained so the message history can later be modeled as a branching
tree/DAG (append = new node; branch = pointer to an earlier node). That modeling is not implemented;
this feature ships only the transport and storage plumbing.

## CLI surface

`ninfer-serve`:

- `--persist-endpoint HOST:PORT` — enable persistence and set the remote target (empty/unset =>
  disabled).
- `--persist-queue-capacity N` — bounded in-memory queue size (default 4096; positive).
- `--persist-reconnect-ms N` — backoff between reconnect attempts (default 1000; positive).

`run-tp4.sh`:

- `--remote-req-log[=HOST:PORT]` — ensure the Python receiver is running (spawning it detached when
  its port is not already listening), store the DB under `/media/optane/reqlogs`, and forward
  `--persist-endpoint` to `ninfer-serve`. Default endpoint `127.0.0.1:9100`. Overridable with
  `NINFER_REMOTE_REQ_LOG_ENDPOINT`, `NINFER_REMOTE_REQ_LOG_DIR`, `NINFER_REMOTE_REQ_LOG_DB`, and
  `NINFER_REMOTE_REQ_LOG_PYTHON`.

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

## Verification

- `ninfer_request_persist_test`: codec golden bytes, round-trip, and reject cases — ok.
- `ninfer_serve_options_test`: new-flag defaults, parsing, invalid values, and usage — ok.
- Python `pytest tests/test_request_persist.py`: 9 passed — golden bytes matching the C++ vector,
  decode/round-trip, frame reader, field extraction, DB insert/query, and a real TCP loopback
  end-to-end into SQLite.
- GPU end-to-end (real engine, qwen3.8-27b, TP2): two `/v1/chat/completions` requests (1-message and
  3-message history) completed; both rows landed in SQLite with the exact raw bodies
  (`message_count` 1 then 3); SIGTERM flushed the persistor and the receiver logged
  `connection ... closed (2 records)`.
