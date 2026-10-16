"""Asyncio TCP receiver for persisted OpenAI requests.

ninfer-serve ships a copy of each parsed ``/v1/chat/completions`` request to this
server over a length-prefixed binary protocol (see docs/maintainer/request-persistence.md). This module reads
those frames, extracts the JSON body, and stores each record in SQLite.

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
import signal
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

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
    received_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_requests_model   ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_unix_ms);
"""


def init_db(path: str) -> sqlite3.Connection:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
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


def insert_record(conn: sqlite3.Connection, rec: ParsedRecord) -> None:
    message_count, stream = extract_fields(rec.body)
    received_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO requests (request_id, created_unix_ms, protocol, model,"
        " body_json, message_count, stream, received_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rec.request_id, rec.created_unix_ms, rec.protocol, rec.model, rec.body,
         message_count, stream, received_at),
    )


_SHUTDOWN = object()


class PersistServer:
    """Accepts framed request records over TCP and persists them to SQLite.

    All inserts are funneled through a single writer coroutine so SQLite sees one
    serial writer regardless of how many client connections are active.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9100,
                 db_path: str = "out/request_persist.sqlite",
                 logger: Optional[logging.Logger] = None):
        self.host = host
        self.port = port
        self.db_path = db_path
        self.log = logger or logging.getLogger("request_persist")
        self._server: Optional[asyncio.AbstractServer] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._write_queue: Optional[asyncio.Queue] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._handler_tasks = set()
        self._running = False

    @property
    def bound_port(self) -> int:
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self.port

    async def start(self) -> None:
        self._conn = init_db(self.db_path)
        self._write_queue = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._run_writer())
        self._server = await asyncio.start_server(self._on_connect, self.host, self.port)
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
                insert_record(self._conn, rec)
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
        self._conn.close()
        self.log.info("stopped; db at %s", self.db_path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive persisted OpenAI requests over TCP and store them in SQLite.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--db", default="out/request_persist.sqlite")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("request_persist")
    server = PersistServer(host=args.host, port=args.port, db_path=args.db, logger=log)

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