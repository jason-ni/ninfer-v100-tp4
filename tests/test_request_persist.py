from __future__ import annotations

import asyncio
import os
import socket
import sqlite3
import tempfile

import pytest

from tools.request_persist import request_persist_server as rps


# Pinned identically to the C++ sender's unit test (tests/test_request_persist.cpp).
GOLDEN_HEX = (
    "0100000000000000002a00000192cc091400000000176f70656e61695f636861745f636f6d706c6574"
    "696f6e73000000117177656e332e382d3237622f6e76667034000000397b226d6f64656c223a226d"
    "222c226d65737361676573223a5b7b22726f6c65223a2275736572222c22636f6e74656e74223a2268"
    "69227d5d7d"
)
GOLDEN_BODY = '{"model":"m","messages":[{"role":"user","content":"hi"}]}'


def test_golden_bytes_match_cpp() -> None:
    env = rps.encode_envelope(1, 42, 1730000000000, "openai_chat_completions",
                              "qwen3.8-27b/nvfp4", GOLDEN_BODY)
    assert env.hex() == GOLDEN_HEX


def test_decode_golden() -> None:
    rec = rps.decode_envelope(bytes.fromhex(GOLDEN_HEX))
    assert rec.version == 1
    assert rec.request_id == 42
    assert rec.created_unix_ms == 1730000000000
    assert rec.protocol == "openai_chat_completions"
    assert rec.model == "qwen3.8-27b/nvfp4"
    assert rec.body == GOLDEN_BODY


def test_roundtrip_edge_values() -> None:
    body = '{"messages":[{"role":"user","content":"héllo 世界"}]}'
    env = rps.encode_envelope(1, 2**64 - 1, -5, "openai_chat_completions", "m", body)
    rec = rps.decode_envelope(env)
    assert rec.request_id == 2**64 - 1
    assert rec.created_unix_ms == -5
    assert rec.body == body


def test_bad_version_rejected() -> None:
    env = bytearray(rps.encode_envelope(1, 1, 1, "p", "m", "{}"))
    env[0] = 2
    with pytest.raises(ValueError):
        rps.decode_envelope(bytes(env))


def test_truncated_rejected() -> None:
    env = rps.encode_envelope(1, 1, 1, "p", "m", "{}")
    with pytest.raises(ValueError):
        rps.decode_envelope(env[:-1])


def test_read_frame_multiple_and_eof() -> None:
    f1 = rps.frame(rps.encode_envelope(1, 1, 1000, "p", "m", '{"a":1}'))
    f2 = rps.frame(rps.encode_envelope(1, 2, 2000, "p", "m", '{"b":2}'))

    async def go():
        reader = asyncio.StreamReader()
        reader.feed_data(f1 + f2)
        reader.feed_eof()
        return await rps.read_frame(reader), await rps.read_frame(reader), \
            await rps.read_frame(reader)

    r1, r2, r3 = asyncio.run(go())
    assert r1.request_id == 1 and r1.body == '{"a":1}'
    assert r2.request_id == 2 and r2.body == '{"b":2}'
    assert r3 is None


def test_extract_fields() -> None:
    assert rps.extract_fields('{"messages":[{"role":"u"},{"role":"a"}],"stream":true}') == (2, 1)
    assert rps.extract_fields('{"messages":[],"stream":false}') == (0, 0)
    assert rps.extract_fields('{"nope":1}') == (None, None)
    assert rps.extract_fields("not json") == (None, None)


def test_db_insert_and_query() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "x.sqlite")
        conn = rps.init_db(db)
        rec = rps.ParsedRecord(1, 7, 123, "openai_chat_completions", "model-x",
                               '{"messages":[{"role":"user","content":"hi"}],"stream":true}')
        rps.insert_record(conn, rec)
        conn.commit()
        row = conn.execute(
            "SELECT request_id, created_unix_ms, protocol, model, body_json,"
            " message_count, stream FROM requests").fetchone()
        conn.close()
    assert row == (7, 123, "openai_chat_completions", "model-x",
                   '{"messages":[{"role":"user","content":"hi"}],"stream":true}', 1, 1)


def test_end_to_end_loopback() -> None:
    # Full transport path without a GPU: real TCP connection -> server -> SQLite.
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "e2e.sqlite")

        async def go():
            server = rps.PersistServer(host="127.0.0.1", port=0, db_path=db)
            await server.start()
            port = server.bound_port
            try:
                loop = asyncio.get_running_loop()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                await loop.sock_connect(sock, ("127.0.0.1", port))
                payload = (
                    rps.frame(rps.encode_envelope(1, 100, 111, "openai_chat_completions",
                                                  "m1", '{"messages":[{"role":"user"}]}'))
                    + rps.frame(rps.encode_envelope(1, 101, 222, "openai_chat_completions",
                                                    "m1", '{"messages":[{"role":"a"}]}')))
                await loop.sock_sendall(sock, payload)
                sock.close()
                await asyncio.sleep(0.2)  # let the writer task flush both records
            finally:
                await server.stop()

        asyncio.run(go())

        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT request_id, model, body_json FROM requests ORDER BY request_id").fetchall()
        conn.close()
    assert [r[0] for r in rows] == [100, 101]
    assert rows[0][2] == '{"messages":[{"role":"user"}]}'
    assert rows[1][2] == '{"messages":[{"role":"a"}]}'