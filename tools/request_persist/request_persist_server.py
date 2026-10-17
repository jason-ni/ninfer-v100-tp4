"""Asyncio TCP receiver for persisted OpenAI requests.

ninfer-serve ships a copy of each parsed ``/v1/chat/completions`` request to this
server over a length-prefixed binary protocol (see docs/maintainer/request-persistence.md). This module reads
those frames, extracts the JSON body, and stores each record in SQLite. A stdlib
``http.server`` REST admin interface on a separate port exposes query/delete/vacuum over the stored
records (``GET /health``/``/stats``/``/requests``, ``DELETE /requests``, ``POST /vacuum``).

The wire codec (``encode_envelope`` / ``decode_envelope``) is transport-agnostic and
is asserted against the same golden byte vector as the C++ sender's unit test, so the
cross-language contract is pinned on both sides.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import struct
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

PROTOCOL_VERSION = 1

# Envelope layout (big-endian): version(1) flags(1) request_id(u64) created_unix_ms(i64)
# then three length-prefixed UTF-8 strings: protocol, model, body.
_HEADER = struct.Struct(">BBQq")
_LEN_PREFIX = struct.Struct(">I")


@dataclass
class ParsedRecord:
    version: int
    request_id: int
    created_unix_ms: int
    protocol: str
    model: str
    body: str  # raw JSON body text, unmodified


def encode_envelope(version: int, request_id: int, created_unix_ms: int,
                    protocol: str, model: str, body) -> bytes:
    """Encode a single envelope (no length prefix). Mirrors the C++ encoder."""
    p = protocol.encode("utf-8") if isinstance(protocol, str) else protocol
    m = model.encode("utf-8") if isinstance(model, str) else model
    b = body.encode("utf-8") if isinstance(body, str) else body
    return (_HEADER.pack(version, 0, request_id, created_unix_ms)
            + _LEN_PREFIX.pack(len(p)) + p
            + _LEN_PREFIX.pack(len(m)) + m
            + _LEN_PREFIX.pack(len(b)) + b)


def frame(envelope: bytes) -> bytes:
    """Wrap an envelope with its u32be length prefix for the wire."""
    return _LEN_PREFIX.pack(len(envelope)) + envelope


def decode_envelope(data: bytes) -> ParsedRecord:
    """Parse an envelope (the bytes after the length prefix). Raises ValueError on
    a malformed or unknown-version envelope."""
    if len(data) < _HEADER.size:
        raise ValueError(f"envelope too short: {len(data)} < {_HEADER.size}")
    version, _flags, request_id, created_unix_ms = _HEADER.unpack_from(data, 0)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    off = _HEADER.size

    def read_str() -> str:
        nonlocal off
        (n,) = _LEN_PREFIX.unpack_from(data, off)
        off += 4
        if off + n > len(data):
            raise ValueError("truncated string field")
        raw = data[off:off + n]
        off += n
        return raw.decode("utf-8")

    return ParsedRecord(
        version=version,
        request_id=request_id,
        created_unix_ms=created_unix_ms,
        protocol=read_str(),
        model=read_str(),
        body=read_str(),
    )


async def read_frame(reader: asyncio.StreamReader) -> Optional[ParsedRecord]:
    """Read one length-prefixed frame. Returns None on clean EOF at a frame boundary;
    raises asyncio.IncompleteReadError when the stream ends mid-frame and ValueError on a
    bad envelope."""
    try:
        header = await reader.readexactly(_LEN_PREFIX.size)
    except asyncio.IncompleteReadError:
        return None
    (length,) = _LEN_PREFIX.unpack(header)
    envelope = await reader.readexactly(length)
    return decode_envelope(envelope)


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      INTEGER NOT NULL,
    created_unix_ms INTEGER NOT NULL,
    protocol        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    body_json       TEXT    NOT NULL,
    message_count   INTEGER,
    stream          INTEGER,
    received_at     TEXT    NOT NULL,
    body_truncated  INTEGER NOT NULL DEFAULT 0,
    body_orig_bytes INTEGER
);
CREATE INDEX IF NOT EXISTS idx_requests_model   ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_unix_ms);
"""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add the truncation-tracking columns to databases created before they existed.

    ``ADD COLUMN`` is a schema-only change in SQLite (no row rewrite), so this is cheap even on a
    large existing database and runs idempotently on every startup."""
    have = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    if "body_truncated" not in have:
        conn.execute("ALTER TABLE requests ADD COLUMN body_truncated INTEGER NOT NULL DEFAULT 0")
    if "body_orig_bytes" not in have:
        conn.execute("ALTER TABLE requests ADD COLUMN body_orig_bytes INTEGER")


def init_db(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path, timeout=_DB_CONNECT_TIMEOUT_S)
    # The connect timeout doubles as the busy timeout: it absorbs the brief lock wait when a
    # REST read/delete contends with the single serial insert writer (per-record commits). The
    # DB keeps its existing journal mode (rollback); VACUUM reclaims space by rewriting the main
    # file, which is what the "freed bytes" reporting measures.
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    conn.commit()
    return conn


def extract_fields(body: str):
    """Return (message_count, stream) extracted from an OpenAI chat body, or
    (None, None) when the body is not a matching object."""
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(obj, dict):
        return None, None
    messages = obj.get("messages")
    message_count = len(messages) if isinstance(messages, list) else None
    stream_raw = obj.get("stream")
    stream = 1 if stream_raw is True else (0 if stream_raw is False else None)
    return message_count, stream


def insert_record(conn: sqlite3.Connection, rec: ParsedRecord,
                  max_body_bytes: int = 0) -> None:
    """Insert one record. ``message_count``/``stream`` are extracted from the full body first;
    then, when ``max_body_bytes`` > 0 and the body is larger, the stored body is cut to that many
    bytes on a UTF-8 boundary and flagged (body_truncated=1). The original size is always recorded
    in body_orig_bytes."""
    message_count, stream = extract_fields(rec.body)
    received_at = datetime.now(timezone.utc).isoformat()
    raw = rec.body.encode("utf-8") if isinstance(rec.body, str) else bytes(rec.body)
    orig_bytes = len(raw)
    if max_body_bytes > 0 and orig_bytes > max_body_bytes:
        stored = raw[:max_body_bytes].decode("utf-8", "ignore")
        truncated = 1
    else:
        stored = rec.body if isinstance(rec.body, str) else raw.decode("utf-8", "replace")
        truncated = 0
    conn.execute(
        "INSERT INTO requests (request_id, created_unix_ms, protocol, model,"
        " body_json, message_count, stream, received_at, body_truncated, body_orig_bytes)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rec.request_id, rec.created_unix_ms, rec.protocol, rec.model, stored,
         message_count, stream, received_at, truncated, orig_bytes),
    )


# ---------------------------------------------------------------------------
# REST admin API (stdlib http.server + sqlite3). The persistent insert path
# (TCP -> writer coroutine) is untouched; each HTTP handler opens its own
# short-lived connection so SQLite sees one serial writer plus ad-hoc
# readers/deleters, coordinated by WAL + busy timeout. All query/delete logic
# lives in the pure functions below so it is unit-testable without a server.
# ---------------------------------------------------------------------------

# Stored `created_unix_ms` values below this magnitude are treated as whole
# seconds; at/above it they are milliseconds. 1e12 ms == 2001-09-09 and modern
# wall clocks are ~1.8e9 s / ~1.8e12 ms, so this cleanly disambiguates the two
# units for any 21st-century timestamp regardless of what a producer wrote.
_EPOCH_MS_FLOOR = 10 ** 12
_INT_RE = re.compile(r"^[+-]?\d+$")
_FALSY_FLAGS = ("0", "false", "no")
_DB_CONNECT_TIMEOUT_S = 30.0


def infer_time_unit(max_value) -> str:
    """Return 's' or 'ms' for a stored epoch column from its maximum value."""
    if max_value is None:
        return "s"
    return "ms" if int(max_value) >= _EPOCH_MS_FLOOR else "s"


def now_epoch(unit: str) -> int:
    """Current wall-clock time in the given unit ('s' or 'ms')."""
    secs = time.time()
    return int(secs) if unit == "s" else int(secs * 1000)


def _to_unit(epoch_seconds: float, unit: str) -> int:
    return int(epoch_seconds) if unit == "s" else int(round(epoch_seconds * 1000))


def parse_epoch_bound(text: str, unit: str) -> int:
    """Parse a time bound into the stored column's unit.

    Accepts a bare integer (assumed already in `unit`) or an ISO-8601 date /
    datetime (naive => UTC; trailing 'Z' or an explicit offset is honored).
    Raises ValueError on anything else.
    """
    text = text.strip()
    if _INT_RE.match(text):
        return int(text)
    iso = text[:-1] + "+00:00" if text[-1:] in ("Z", "z") else text
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _to_unit(dt.timestamp(), unit)


def parse_ids(text: str):
    """Parse a comma-separated list of integer ids."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if not _INT_RE.match(part):
            raise ValueError(f"invalid id: {part!r}")
        out.append(int(part))
    if not out:
        raise ValueError("empty id list")
    return out


def build_request_filter(params: dict, unit: str):
    """Build (where_sql, args, has_criteria) from a whitelisted single-value dict.

    Unknown keys are ignored. Time bounds (`before`/`after`, `older_than_days`,
    `newer_than_days`) are normalized into the stored column's `unit`. Returns an
    empty where clause and has_criteria=False when no criterion is supplied.
    Raises ValueError on malformed numeric parameters.
    """
    clauses, args = [], []

    def add(sql: str, *vals) -> None:
        clauses.append(sql)
        args.extend(vals)

    if params.get("model"):
        add("model = ?", params["model"])
    if params.get("protocol"):
        add("protocol = ?", params["protocol"])
    if params.get("stream") in ("0", "1"):
        add("stream = ?", int(params["stream"]))
    if params.get("min_messages"):
        add("message_count IS NOT NULL AND message_count >= ?",
            int(params["min_messages"]))
    if params.get("max_messages"):
        add("message_count IS NOT NULL AND message_count <= ?",
            int(params["max_messages"]))
    if params.get("min_body_kb"):
        add("length(body_json) >= ?", int(float(params["min_body_kb"]) * 1024))
    if params.get("max_body_kb"):
        add("length(body_json) <= ?", int(float(params["max_body_kb"]) * 1024))
    if params.get("id"):
        add("id = ?", int(params["id"]))
    if params.get("ids"):
        ids = parse_ids(params["ids"])
        add("id IN (" + ",".join("?" * len(ids)) + ")", *ids)
    if params.get("before"):
        add("created_unix_ms < ?", parse_epoch_bound(params["before"], unit))
    if params.get("after"):
        add("created_unix_ms > ?", parse_epoch_bound(params["after"], unit))
    for key, op in (("older_than_days", "<"), ("newer_than_days", ">")):
        if params.get(key):
            days = float(params[key])
            step = 86400 * (1000 if unit == "ms" else 1)
            add(f"created_unix_ms {op} ?", now_epoch(unit) - int(days * step))

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, args, bool(clauses)


def _flag(params: dict, key: str, default: bool = False) -> bool:
    """A boolean flag: true when the key is present and not explicitly falsy, so a
    bare `?flag` (empty value) counts as set."""
    if key not in params:
        return default
    return params[key].strip().lower() not in _FALSY_FLAGS


def _db_file_bytes(db_path: str) -> int:
    """On-disk size of the database file (what VACUUM reclaims). Transient -wal/-shm
    sidecar files are excluded: they are runtime artifacts, not stored data."""
    return os.path.getsize(db_path) if os.path.exists(db_path) else 0


def row_to_dict(row, include_body: bool) -> dict:
    (rec_id, request_id, created, protocol, model, body, message_count, stream,
     received_at, body_len, body_truncated, body_orig_bytes) = row
    d = {
        "id": rec_id,
        "request_id": request_id,
        "created_unix_ms": created,
        "protocol": protocol,
        "model": model,
        "message_count": message_count,
        "stream": stream,
        "received_at": received_at,
        # SQLite length() is a character count; for the ASCII/base64 chat bodies this
        # store it equals the byte length, which is what drives the on-disk size.
        "body_bytes": body_len,
        "body_truncated": bool(body_truncated),
        "body_orig_bytes": body_orig_bytes,
    }
    if include_body and body is not None:
        d["body"] = body
    return d


def _stored_unit(conn: sqlite3.Connection) -> str:
    v = conn.execute("SELECT MAX(created_unix_ms) FROM requests").fetchone()[0]
    return infer_time_unit(v)


def _open_db(db_path: str):
    """A context manager yielding a fresh read/write connection with a long busy
    timeout; closes (and rolls back any uncommitted change) on exit."""
    return closing(sqlite3.connect(db_path, timeout=_DB_CONNECT_TIMEOUT_S))


def query_requests(conn: sqlite3.Connection, params: dict) -> dict:
    unit = _stored_unit(conn)
    where, args, _ = build_request_filter(params, unit)
    include_body = _flag(params, "include_body", default=True)
    limit = max(1, min(int(params.get("limit", "50")), 1000))
    offset = max(0, int(params.get("offset", "0")))
    order = "id ASC" if params.get("order", "desc").strip().lower() == "asc" else "id DESC"
    total = conn.execute("SELECT COUNT(*) FROM requests" + where, args).fetchone()[0]
    body_col = "body_json" if include_body else "NULL"
    sql = ("SELECT id, request_id, created_unix_ms, protocol, model, "
           f"{body_col} AS body, message_count, stream, received_at, "
           f"length(body_json) AS body_len, body_truncated, body_orig_bytes "
           f"FROM requests{where} ORDER BY {order} LIMIT ? OFFSET ?")
    rows = conn.execute(sql, args + [limit, offset]).fetchall()
    return {"total": total, "offset": offset, "limit": limit, "order": order,
            "rows": [row_to_dict(r, include_body) for r in rows]}


def get_request(conn: sqlite3.Connection, row_id: int):
    row = conn.execute(
        "SELECT id, request_id, created_unix_ms, protocol, model, body_json,"
        " message_count, stream, received_at, length(body_json),"
        " body_truncated, body_orig_bytes FROM requests WHERE id = ?",
        (row_id,)).fetchone()
    return None if row is None else row_to_dict(row, True)


def delete_requests(conn: sqlite3.Connection, db_path: str, params: dict,
                    dry_run: bool, vacuum: bool) -> dict:
    unit = _stored_unit(conn)
    where, args, has_crit = build_request_filter(params, unit)
    is_all = _flag(params, "all")
    if not has_crit and not is_all:
        raise ValueError("no delete criteria supplied (use all=true, ids, before/after,"
                         " older_than_days, model, ...)")
    if dry_run:
        n = conn.execute("SELECT COUNT(*) FROM requests" + where, args).fetchone()[0]
        return {"deleted": 0, "would_delete": n, "dry_run": True, "vacuumed": False}
    bytes_before = _db_file_bytes(db_path)
    cur = conn.execute("DELETE FROM requests" + where, args)
    conn.commit()
    result = {"deleted": cur.rowcount, "dry_run": False}
    if vacuum:
        conn.execute("VACUUM")
        bytes_after = _db_file_bytes(db_path)
        result.update({"vacuumed": True, "bytes_before": bytes_before,
                       "bytes_after": bytes_after, "bytes_freed": bytes_before - bytes_after})
    else:
        result["vacuumed"] = False
    return result


def vacuum_only(conn: sqlite3.Connection, db_path: str) -> dict:
    bytes_before = _db_file_bytes(db_path)
    conn.execute("VACUUM")
    bytes_after = _db_file_bytes(db_path)
    return {"vacuumed": True, "bytes_before": bytes_before,
            "bytes_after": bytes_after, "bytes_freed": bytes_before - bytes_after}


def db_stats(conn: sqlite3.Connection, db_path: str) -> dict:
    rows = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    cmin, cmax = conn.execute(
        "SELECT MIN(created_unix_ms), MAX(created_unix_ms) FROM requests").fetchone()
    rmin, rmax = conn.execute(
        "SELECT MIN(received_at), MAX(received_at) FROM requests").fetchone()
    bl = conn.execute(
        "SELECT MIN(length(body_json)), AVG(length(body_json)),"
        " MAX(length(body_json)), SUM(length(body_json)) FROM requests").fetchone()
    trunc = conn.execute(
        "SELECT SUM(COALESCE(body_truncated, 0)) FROM requests").fetchone()[0]
    by_model = dict(conn.execute(
        "SELECT model, COUNT(*) FROM requests GROUP BY model ORDER BY COUNT(*) DESC")
        .fetchall())
    by_protocol = dict(conn.execute(
        "SELECT protocol, COUNT(*) FROM requests GROUP BY protocol"
        " ORDER BY COUNT(*) DESC").fetchall())
    return {
        "rows": rows,
        "truncated_rows": int(trunc or 0),
        "db_file_bytes": _db_file_bytes(db_path),
        "created_unix_ms": {"min": cmin, "max": cmax, "unit": infer_time_unit(cmax)},
        "received_at": {"min": rmin, "max": rmax},
        "body_chars": {"min": bl[0],
                       "avg": round(bl[1], 1) if bl[1] is not None else None,
                       "max": bl[2], "total": bl[3]},
        "by_model": by_model,
        "by_protocol": by_protocol,
    }


class RequestPersistHttpHandler(BaseHTTPRequestHandler):
    """JSON REST admin interface over the persisted-requests database."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # route the access log to the server logger
        self.server.log.info("%s %s", self.address_string(), format % args)

    def _send_json(self, obj, status: int = 200) -> None:
        payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _split(self):
        parsed = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
        return parsed.path, qs

    def do_GET(self) -> None:
        path, qs = self._split()
        try:
            if path == "/health":
                return self._send_json({
                    "ok": True, "db": self.server.db_path,
                    "tcp_port": self.server.tcp_port, "http_port": self.server.http_port,
                    "version": PROTOCOL_VERSION})
            if path == "/stats":
                with _open_db(self.server.db_path) as conn:
                    return self._send_json(db_stats(conn, self.server.db_path))
            if path == "/requests":
                with _open_db(self.server.db_path) as conn:
                    return self._send_json(query_requests(conn, qs))
            m = re.fullmatch(r"/requests/(\d+)", path)
            if m:
                with _open_db(self.server.db_path) as conn:
                    rec = get_request(conn, int(m.group(1)))
                return self._error(404, "request id not found") if rec is None \
                    else self._send_json(rec)
            return self._error(404, "not found")
        except ValueError as exc:
            return self._error(400, str(exc))
        except sqlite3.Error as exc:
            return self._error(500, f"database error: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, do not drop the connection silently
            self.server.log.exception("GET %s failed", path)
            return self._error(500, str(exc))

    def do_DELETE(self) -> None:
        path, qs = self._split()
        if path != "/requests":
            return self._error(404, "not found")
        try:
            with _open_db(self.server.db_path) as conn:
                result = delete_requests(conn, self.server.db_path, qs,
                                         dry_run=_flag(qs, "dry_run"),
                                         vacuum=_flag(qs, "vacuum"))
            self.server.log.info("DELETE /requests -> %s", result)
            return self._send_json(result)
        except ValueError as exc:
            return self._error(400, str(exc))
        except sqlite3.Error as exc:
            return self._error(500, f"database error: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.server.log.exception("DELETE /requests failed")
            return self._error(500, str(exc))

    def do_POST(self) -> None:
        path, _qs = self._split()
        if path == "/vacuum":
            try:
                with _open_db(self.server.db_path) as conn:
                    result = vacuum_only(conn, self.server.db_path)
                self.server.log.info("POST /vacuum -> %s", result)
                return self._send_json(result)
            except sqlite3.Error as exc:
                return self._error(500, f"database error: {exc}")
            except Exception as exc:  # noqa: BLE001
                self.server.log.exception("POST /vacuum failed")
                return self._error(500, str(exc))
        return self._error(404, "not found")


class _ThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # Populated by PersistServer.start() before serving; read by RequestPersistHttpHandler.
    db_path: str
    tcp_port: int
    http_port: int
    log: logging.Logger


_SHUTDOWN = object()


class PersistServer:
    """Accepts framed request records over TCP and persists them to SQLite.

    All inserts are funneled through a single writer coroutine so SQLite sees one
    serial writer regardless of how many client connections are active.
"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9100,
                  db_path: str = "out/request_persist.sqlite",
                  logger: Optional[logging.Logger] = None,
                  http_host: str = "127.0.0.1", http_port: int = 9101,
                  enable_http: bool = True, max_body_kb: int = 1024):
        self.host = host
        self.port = port
        self.db_path = db_path
        self.http_host = http_host
        # http_port: a positive value binds that port; 0 binds an OS-chosen ephemeral port.
        self.http_port = http_port
        self.enable_http = enable_http
        # Stored request bodies are truncated above this many bytes (0 disables truncation).
        self.max_body_bytes = int(max_body_kb) * 1024
        self.log = logger or logging.getLogger("request_persist")
        self._server: Optional[asyncio.AbstractServer] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._write_queue: Optional[asyncio.Queue] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._handler_tasks = set()
        self._http: Optional[_ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def bound_port(self) -> int:
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self.port

    @property
    def http_bound_port(self) -> int:
        if self._http is not None:
            return self._http.server_address[1]
        return self.http_port

    async def start(self) -> None:
        self._conn = init_db(self.db_path)
        self._write_queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._run_writer())
        self._server = await asyncio.start_server(self._on_connect, self.host, self.port)
        if self.enable_http:
            self._http = _ThreadingHTTPServer((self.http_host, self.http_port),
                                              RequestPersistHttpHandler)
            self._http.db_path = self.db_path
            self._http.tcp_port = self.bound_port
            self._http.http_port = self._http.server_address[1]
            self._http.log = self.log
            self._http_thread = threading.Thread(
                target=self._http.serve_forever, name="rp-rest", daemon=True)
            self._http_thread.start()
            self.log.info("rest api on http://%s:%d (/health /stats /requests)",
                          self.http_host, self._http.http_port)
        self._running = True
        self.log.info("listening on %s:%d -> %s", self.host, self.bound_port, self.db_path)

    async def _on_connect(self, reader: asyncio.StreamReader,
                          writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)
        await self._handle_client(reader, writer)

    async def _handle_client(self, reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        count = 0
        try:
            while True:
                rec = await read_frame(reader)
                if rec is None:
                    break
                await self._write_queue.put(rec)
                count += 1
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as exc:  # malformed frame / bad version: drop the connection
            self.log.warning("connection %s closed: %s", peer, exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            self.log.info("connection %s closed (%d records)", peer, count)

    async def _run_writer(self) -> None:
        while True:
            rec = await self._write_queue.get()
            try:
                if rec is _SHUTDOWN:
                    break
                insert_record(self._conn, rec, self.max_body_bytes)
                self._conn.commit()
            finally:
                self._write_queue.task_done()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        # Close active clients so their handlers observe EOF and finish.
        for task in list(self._handler_tasks):
            task.cancel()
        if self._handler_tasks:
            await asyncio.gather(*list(self._handler_tasks), return_exceptions=True)
        await self._write_queue.put(_SHUTDOWN)
        await self._writer_task
        if self._http is not None:
            self._http.shutdown()  # joins the serve_forever loop
            self._http.server_close()
            if self._http_thread is not None:
                self._http_thread.join(timeout=2.0)
            self._http = None
            self._http_thread = None
        self._conn.close()
        self.log.info("stopped; db at %s", self.db_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive persisted OpenAI requests over TCP and store them in SQLite.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--db", default="out/request_persist.sqlite")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--http-host", default="127.0.0.1",
                        help="REST admin bind address (default 127.0.0.1)")
    parser.add_argument("--http-port", type=int, default=9101,
                        help="REST admin port; 0 binds an ephemeral port")
    parser.add_argument("--no-http", action="store_true",
                        help="disable the REST admin interface entirely")
    parser.add_argument("--max-body-kb", type=int, default=1024,
                        help="truncate stored request bodies above this size in KiB (0 disables)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("request_persist")
    server = PersistServer(host=args.host, port=args.port, db_path=args.db, logger=log,
                            http_host=args.http_host, http_port=args.http_port,
                            enable_http=not args.no_http, max_body_kb=args.max_body_kb)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        log.info("signal received, shutting down")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except (NotImplementedError, RuntimeError):
            pass

    async def _amain() -> None:
        await server.start()
        await stop_event.wait()
        await server.stop()

    try:
        loop.run_until_complete(_amain())
    finally:
        loop.close()


if __name__ == "__main__":
    main()