from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

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
        got = rps.get_request(conn, 1)
        conn.close()
    assert got["request_id"] == 7 and got["created_unix_ms"] == 123
    assert got["protocol"] == "openai_chat_completions" and got["model"] == "model-x"
    assert got["body"] == rec.body          # reconstructed byte-for-byte
    assert got["message_count"] == 1 and got["stream"] == 1
    assert got["body_bytes"] == len(rec.body)


def test_end_to_end_loopback() -> None:
    # Full transport path without a GPU: real TCP connection -> server -> SQLite.
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "e2e.sqlite")

        async def go():
            server = rps.PersistServer(host="127.0.0.1", port=0, db_path=db, http_port=0)
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
        rows = rps.query_requests(conn, {"order": "asc", "limit": "10"})["rows"]
        conn.close()
    assert [r["request_id"] for r in rows] == [100, 101]
    assert rows[0]["body"] == '{"messages":[{"role":"user"}]}'
    assert rows[1]["body"] == '{"messages":[{"role":"a"}]}'


# --- REST admin API --------------------------------------------------------

def _http(method: str, url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _seed_db(path: str, records) -> str:
    conn = rps.init_db(path)
    for rec in records:
        rps.insert_record(conn, rec)
    conn.commit()
    conn.close()
    return path


def test_infer_time_unit() -> None:
    assert rps.infer_time_unit(None) == "s"
    assert rps.infer_time_unit(1_792_219_910) == "s"           # seconds (~2026)
    assert rps.infer_time_unit(1_792_219_910_000) == "ms"      # milliseconds (~2026)
    assert rps.infer_time_unit(999_999_999_999) == "s"         # just under the floor
    assert rps.infer_time_unit(1_000_000_000_000) == "ms"      # at the floor


def test_parse_epoch_bound_integer_passthrough() -> None:
    assert rps.parse_epoch_bound("1792219910", "s") == 1792219910
    assert rps.parse_epoch_bound("-5", "ms") == -5


def test_parse_epoch_bound_iso() -> None:
    assert rps.parse_epoch_bound("2026-10-01", "s") == \
        int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())
    assert rps.parse_epoch_bound("2026-10-01T12:00:00Z", "ms") == \
        int(datetime(2026, 10, 1, 12, tzinfo=timezone.utc).timestamp() * 1000)
    assert rps.parse_epoch_bound("2026-10-01T07:00:00+05:00", "s") == \
        int(datetime(2026, 10, 1, 2, 0, tzinfo=timezone.utc).timestamp())


def test_parse_epoch_bound_invalid() -> None:
    with pytest.raises(ValueError):
        rps.parse_epoch_bound("not-a-date", "s")


def test_parse_ids() -> None:
    assert rps.parse_ids("1, 2,3") == [1, 2, 3]
    assert rps.parse_ids("7") == [7]
    with pytest.raises(ValueError):
        rps.parse_ids("1,x")
    with pytest.raises(ValueError):
        rps.parse_ids("")


def test_build_filter_empty_and_unknown() -> None:
    where, args, has = rps.build_request_filter({"bogus": "x"}, "s")
    assert where == "" and args == [] and has is False


def test_build_filter_combines_with_and() -> None:
    where, args, has = rps.build_request_filter(
        {"model": "m", "stream": "1", "min_messages": "2", "before": "1000",
         "ids": "4,5"}, "s")
    assert has is True
    assert where.count(" AND ") >= 4
    assert "model = ?" in where and "id IN (?,?)" in where and "created_unix_ms < ?" in where
    assert args.count("m") == 1
    assert 1000 in args


def test_build_filter_bad_numeric_raises() -> None:
    with pytest.raises(ValueError):
        rps.build_request_filter({"min_messages": "abc"}, "s")
    with pytest.raises(ValueError):
        rps.build_request_filter({"older_than_days": "soon"}, "s")


def test_flag_semantics() -> None:
    assert rps._flag({"dry_run": ""}, "dry_run") is True       # bare flag counts as set
    assert rps._flag({"dry_run": "true"}, "dry_run") is True
    assert rps._flag({"dry_run": "0"}, "dry_run") is False
    assert rps._flag({}, "all") is False
    assert rps._flag({"include_body": "false"}, "include_body", default=True) is False
    assert rps._flag({}, "include_body", default=True) is True


def test_delete_dry_run_then_commit(tmp_path) -> None:
    now = int(time.time())
    db = str(tmp_path / "d.sqlite")
    _seed_db(db, [
        rps.ParsedRecord(1, 1, now - 20 * 86400, "p", "old", "{}"),
        rps.ParsedRecord(1, 2, now - 3 * 86400, "p", "new", "{}"),
        rps.ParsedRecord(1, 3, now - 1, "p", "new", "{}"),
    ])
    with rps._open_db(db) as c:
        dry = rps.delete_requests(c, db, {"older_than_days": "7"}, True, False)
        assert dry["dry_run"] is True and dry["would_delete"] == 1   # only the 20-day row
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from requests").fetchone()[0] == 3
    conn.close()
    with rps._open_db(db) as c:
        res = rps.delete_requests(c, db, {"older_than_days": "7"}, False, False)
    assert res["deleted"] == 1 and res["vacuumed"] is False
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from requests").fetchone()[0] == 2
    conn.close()


def test_delete_no_criteria_rejected() -> None:
    db = str(tempfile.mkstemp(suffix=".sqlite")[1])
    os.unlink(db)
    _seed_db(db, [rps.ParsedRecord(1, 1, 10, "p", "m", "{}")])
    with pytest.raises(ValueError):
        with rps._open_db(db) as c:
            rps.delete_requests(c, db, {}, False, False)


def test_delete_all_and_vacuum_reports_bytes(tmp_path) -> None:
    db = str(tmp_path / "all.sqlite")
    _seed_db(db, [rps.ParsedRecord(1, i, i, "p", "m", "x" * 4096) for i in range(200)])
    with rps._open_db(db) as c:
        res = rps.delete_requests(c, db, {"all": "true"}, False, True)
    assert res["deleted"] == 200 and res["vacuumed"] is True
    assert res["bytes_before"] > res["bytes_after"] >= 0
    assert res["bytes_freed"] == res["bytes_before"] - res["bytes_after"] > 0
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from requests").fetchone()[0] == 0
    conn.close()


def test_query_pagination_order_and_include_body(tmp_path) -> None:
    db = str(tmp_path / "q.sqlite")
    _seed_db(db, [rps.ParsedRecord(1, i, i, "p", f"m{i}", "{\"messages\":[]}")
                  for i in range(1, 6)])
    with rps._open_db(db) as c:
        page = rps.query_requests(c, {"limit": "2", "order": "asc"})
    assert page["total"] == 5 and len(page["rows"]) == 2
    assert [r["request_id"] for r in page["rows"]] == [1, 2]
    assert "body" in page["rows"][0]
    with rps._open_db(db) as c:
        no_body = rps.query_requests(c, {"include_body": "false", "limit": "10"})
    assert all("body" not in r for r in no_body["rows"])
    assert all(isinstance(r["body_bytes"], int) for r in no_body["rows"])


def test_query_filter_by_model(tmp_path) -> None:
    db = str(tmp_path / "mf.sqlite")
    _seed_db(db, [rps.ParsedRecord(1, i, i, "p", ("a" if i % 2 else "b"), "{}")
                  for i in range(1, 7)])
    with rps._open_db(db) as c:
        only_a = rps.query_requests(c, {"model": "a", "limit": "50"})
    assert only_a["total"] == 3 and all(r["model"] == "a" for r in only_a["rows"])


def test_get_request_found_and_missing(tmp_path) -> None:
    db = str(tmp_path / "g.sqlite")
    _seed_db(db, [rps.ParsedRecord(1, 99, 5, "p", "m", "{\"k\":1}")])
    with rps._open_db(db) as c:
        got = rps.get_request(c, 1)
        missing = rps.get_request(c, 424242)
    assert got["request_id"] == 99 and got["body"] == "{\"k\":1}"
    assert missing is None


def test_stats_shape(tmp_path) -> None:
    db = str(tmp_path / "s.sqlite")
    now = int(time.time())
    _seed_db(db, [
        rps.ParsedRecord(1, 1, now - 100, "p", "m1", "aaaa"),
        rps.ParsedRecord(1, 2, now - 50, "p", "m2", "bbbbcc"),
    ])
    with rps._open_db(db) as c:
        s = rps.db_stats(c, db)
    assert s["rows"] == 2 and s["created_unix_ms"]["unit"] == "s"
    assert s["by_model"] == {"m1": 1, "m2": 1} or set(s["by_model"]) == {"m1", "m2"}
    assert s["body_chars"]["max"] == 6 and s["body_chars"]["total"] == 10
    assert s["db_file_bytes"] > 0


def test_rest_serves_ui(tmp_path) -> None:
    db = str(tmp_path / "ui.sqlite")
    out = {}

    async def go():
        server = rps.PersistServer(host="127.0.0.1", port=0, db_path=db,
                                   http_host="127.0.0.1", http_port=0)
        await server.start()
        base = f"http://127.0.0.1:{server.http_bound_port}"
        try:
            for path in ("/", "/index.html"):
                with urllib.request.urlopen(base + path, timeout=5.0) as resp:
                    out[path] = (resp.status, resp.headers.get("Content-Type"),
                                 resp.read().decode("utf-8"))
        finally:
            await server.stop()

    asyncio.run(go())

    for path in ("/", "/index.html"):
        status, ctype, text = out[path]
        assert status == 200
        assert (ctype or "").startswith("text/html")
        # Self-contained page marker plus the fetch-driven client and the routes it calls.
        assert "ninfer-request-persist-ui" in text
        assert "fetch(" in text
        assert "/requests" in text


def test_rest_end_to_end(tmp_path) -> None:
    db = str(tmp_path / "rest.sqlite")
    now = int(time.time())

    def mk(rid, age_days):
        return rps.ParsedRecord(1, rid, now - age_days * 86400,
                                "openai_chat_completions", "qwen3.8-27b",
                                '{"messages":[{"role":"user","content":"hi"}]}')

    recs = [mk(1, 30), mk(2, 10), mk(3, 1)]
    # rec 3 carries a large (>1 KiB) unique body: it must round-trip losslessly.
    recs[2] = rps.ParsedRecord(
        1, 3, now - 86400, "openai_chat_completions", "qwen3.8-27b",
        '{"messages":[{"role":"user","content":"' + "y" * 2000 + '"}]}')
    out = {}

    async def go():
        server = rps.PersistServer(host="127.0.0.1", port=0, db_path=db,
                                   http_host="127.0.0.1", http_port=0)
        await server.start()
        base = f"http://127.0.0.1:{server.http_bound_port}"
        try:
            loop = asyncio.get_running_loop()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            await loop.sock_connect(sock, ("127.0.0.1", server.bound_port))
            payload = b"".join(
                rps.frame(rps.encode_envelope(1, r.request_id, r.created_unix_ms,
                                               r.protocol, r.model, r.body))
                for r in recs)
            await loop.sock_sendall(sock, payload)
            sock.close()
            await asyncio.sleep(0.3)  # let the writer coroutine flush all records

            out["health"] = _http("GET", base + "/health")
            out["stats"] = _http("GET", base + "/stats")
            out["list"] = _http("GET", base + "/requests?limit=10&order=asc")
            newest_id = next(r["id"] for r in out["list"][1]["rows"]
                             if r["created_unix_ms"] == now - 86400)
            out["one"] = _http("GET", f"{base}/requests/{newest_id}")
            out["missing"] = _http("GET", f"{base}/requests/999999")
            out["bad"] = _http("GET", base + "/requests?limit=abc")
            out["dry"] = _http("DELETE", base + "/requests?older_than_days=7&dry_run")
            out["delete"] = _http("DELETE", base + "/requests?older_than_days=7")
            out["stats_after"] = _http("GET", base + "/stats")
            out["vacuum"] = _http("POST", base + "/vacuum")
        finally:
            await server.stop()

    asyncio.run(go())

    assert out["health"][0] == 200 and out["health"][1]["ok"] is True
    assert out["stats"][0] == 200 and out["stats"][1]["rows"] == 3
    assert out["stats"][1]["created_unix_ms"]["unit"] == "s"
    assert len(out["list"][1]["rows"]) == 3
    assert out["one"][0] == 200 and "body" in out["one"][1]
    assert out["missing"][0] == 404
    assert out["bad"][0] == 400
    # 30-day and 10-day rows are older than 7 days; the 1-day row is not.
    assert out["dry"][0] == 200 and out["dry"][1]["dry_run"] is True \
        and out["dry"][1]["would_delete"] == 2
    assert out["delete"][0] == 200 and out["delete"][1]["deleted"] == 2
    assert out["stats_after"][1]["rows"] == 1
    assert out["vacuum"][0] == 200 and out["vacuum"][1]["vacuumed"] is True

    # Bodies are stored losslessly: the >1 KiB unique body reconstructs byte-for-byte and its
    # reconstructed size is reported as body_bytes.
    big_row = next(r for r in out["list"][1]["rows"] if r["request_id"] == 3)
    assert big_row["message_count"] == 1 and big_row["body_bytes"] > 1024
    assert out["stats"][1]["unique_chunks"] >= 1 and out["stats"][1]["chunk_bytes"] > 0

    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from requests").fetchone()[0] == 1
    conn.close()


def test_split_body_is_byte_exact() -> None:
    bodies = [
        '{"model":"m","messages":[{"role":"user","content":"hi"}]}',
        '{"messages":[]}',
        '{"a":1}',                                    # no messages key -> whole body is head
        'not json',                                   # unparsable -> opaque
        '{"messages":[{"role":"system","content":"中"},{"role":"user","content":"hi 世界"}]}',
        '{"model":"m","temperature":0.5,"messages": [ {"role":"user","content":"x"} ,'
        ' {"role":"assistant","content":"y"} ], "stream":true}',  # irregular whitespace
        '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},'
        '{"role":"user","content":"c"}]}',
    ]
    for body in bodies:
        head, parts, tail = rps.split_body(body)
        assert head + b"".join(parts) + tail == body.encode("utf-8")


def test_split_body_appends_do_not_rewrite_earlier_parts() -> None:
    # Appending a message must leave every earlier span (and the tail) byte-identical, which is
    # what makes an agent's growing history dedup to a handful of chunks.
    def body(nmsgs):
        msgs = "".join('{"role":"user","content":"m%d"},' % i for i in range(nmsgs))
        msgs = msgs.rstrip(",")
        return '{"model":"m","messages":[' + msgs + "]}"

    prev_parts = None
    for n in range(1, 6):
        head, parts, tail = rps.split_body(body(n))
        if prev_parts is not None:
            assert parts[:len(prev_parts)] == prev_parts
        prev_parts = parts


def test_pack_unpack_ids_roundtrip() -> None:
    for ids in ([], [0], [1, 2, 3], [127, 128, 300, 2 ** 40], list(range(500))):
        assert rps.unpack_ids(rps.pack_ids(ids)) == list(ids)


def test_content_addressing_dedups_shared_history(tmp_path) -> None:
    db = str(tmp_path / "cas.sqlite")
    conn = rps.init_db(db)
    def mk(rid, msgs):
        return rps.ParsedRecord(1, rid, 1000 + rid, "openai_chat_completions", "m",
                                json.dumps({"model": "m", "tools": [{"x": 1}], "messages": msgs}))
    recs = [
        mk(1, [{"role": "system", "content": "sys"}]),
        mk(2, [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]),
        mk(3, [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"},
               {"role": "assistant", "content": "yo"}]),
    ]
    for rec in recs:
        rps.insert_record(conn, rec)
    conn.commit()
    # head (tools framing), tail, and the three distinct messages -> 5 unique chunks, not 15.
    assert conn.execute("select count(*) from chunks").fetchone()[0] == 5
    for rec in recs:
        got = rps.get_request(conn, rec.request_id)
        assert got["body"] == rec.body
    conn.close()


def test_delete_refcounts_and_gcs_orphan_chunks(tmp_path) -> None:
    db = str(tmp_path / "gc.sqlite")
    conn = rps.init_db(db)
    def mk(rid, msgs):
        return rps.ParsedRecord(1, rid, 1000 + rid, "p", "m",
                                json.dumps({"messages": msgs}))
    rps.insert_record(conn, mk(1, [{"role": "system", "content": "s"}]))
    rps.insert_record(conn, mk(2, [{"role": "system", "content": "s"},
                                   {"role": "user", "content": "u"}]))
    conn.commit()
    total = conn.execute("select count(*) from chunks").fetchone()[0]
    conn.close()
    # Deleting request 2 frees only its unique "u" chunk; the shared system/framing chunks stay.
    with rps._open_db(db) as c:
        out = rps.delete_requests(c, db, {"ids": "2"}, False, False)
    assert out["deleted"] == 1
    conn = sqlite3.connect(db)
    left = conn.execute("select count(*) from chunks").fetchone()[0]
    bodies = rps.query_requests(conn, {"order": "asc"})["rows"]
    conn.close()
    assert left == total - 1
    assert bodies[0]["body"] == mk(1, [{"role": "system", "content": "s"}]).body


def test_init_db_migrates_legacy_body_json_schema(tmp_path) -> None:
    # A database created by the previous verbatim-body schema must be rebuilt into chunks on open,
    # reconstructing the original bodies.
    db = str(tmp_path / "legacy.sqlite")
    body = '{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " request_id INTEGER NOT NULL, created_unix_ms INTEGER NOT NULL,"
        " protocol TEXT NOT NULL, model TEXT NOT NULL, body_json TEXT NOT NULL,"
        " message_count INTEGER, stream INTEGER, received_at TEXT NOT NULL,"
        " body_truncated INTEGER NOT NULL DEFAULT 0, body_orig_bytes INTEGER);")
    conn.execute("INSERT INTO requests (request_id, created_unix_ms, protocol, model,"
                 " body_json, message_count, stream, received_at) VALUES (7, 42, 'p', 'm', ?,"
                 " 1, 1, '2026-01-01T00:00:00+00:00')", (body,))
    conn.commit()
    conn.close()
    conn = rps.init_db(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    got = rps.get_request(conn, 1)
    conn.close()
    assert "body_json" not in cols and {"head_id", "tail_id", "msg_ids"} <= cols
    assert got["request_id"] == 7 and got["created_unix_ms"] == 42
    # message_count/stream are re-derived from the body (the body here has no "stream").
    assert got["body"] == body and got["message_count"] == 1 and got["stream"] is None