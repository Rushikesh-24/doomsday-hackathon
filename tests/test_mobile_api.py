"""Mobile-app routes: map list / offline packs / plan image / registry, write token, Groq assistant (mocked)."""
from __future__ import annotations

import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


def _completed_map(rid: str) -> str:
    """The metric L-walk from test_places, registered as a completed job with places."""
    from tests.test_places import _l_walk
    from vision.navigation import places as PL
    from vision.spatial import jobs as J
    _l_walk(rid)
    import sqlite3
    with sqlite3.connect(J.root() / "spatial.db") as c:
        c.executescript(J._SCHEMA)
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?)", (rid, "east_wing_walk.mp4", "x", "{}", J.now()))
        c.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)", (rid, rid, "completed", "{}", "{}", J.now(), J.now(), None))
    d = J.job_dir(rid)
    if not PL.list_places(d):
        PL.add_place(d, {"name": "Main Entrance", "kind": "entrance", "position": [0, 0, 0], "is_start": True})
        PL.add_place(d, {"name": "CG Lab", "kind": "lab", "position": [10, 0, 6], "aliases": ["graphics lab"]})
        PL.add_place(d, {"name": "Old stairs", "kind": "stairs", "position": [5, 0, 0]})
    return rid


def test_maps_list_pack_etag_and_plan(client):
    rid = _completed_map("mobilemap")
    m = next(x for x in client.get("/maps").json()["maps"] if x["id"] == rid)
    assert m["name"] == "East Wing Walk" and m["places"] == 3 and len(m["version"]) == 16
    r = client.get(f"/maps/{rid}/pack")
    assert r.status_code == 200 and r.headers["etag"] == f'"{m["version"]}"'
    p = r.json()
    assert p["format"] == 2 and p["units"] == "m" and len(p["path"]) == 17
    assert {n["kind"] for n in p["nodes"]} == {"trajectory", "facility", "place"}
    assert p["profiles"]["wheelchair"]["stairs"] is None                      # inf travels as null
    assert any("stairs" in e["tags"] for e in p["edges"])                      # tagged place marks the path
    assert set(p["help"]) == {"wheelchair", "low_vision", "standard"} and p["plan"]["width"] > 0
    assert client.get(f"/maps/{rid}/pack", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    png = client.get(p["plan"]["image"])
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    # editing a place changes the version
    from vision.navigation import places as PL
    from vision.spatial import jobs as J
    pid = PL.list_places(J.job_dir(rid))[1]["id"]
    PL.update_place(J.job_dir(rid), pid, {"description": "Room 204"})
    assert client.get(f"/maps/{rid}/pack").headers["etag"] != r.headers["etag"]


def test_map_registry_rename_and_hide(client):
    rid = _completed_map("mobilemap2")
    assert client.patch(f"/maps/{rid}", json={"name": " "}).status_code == 400
    assert client.patch("/maps/nope", json={"name": "x"}).status_code == 404
    assert client.patch(f"/maps/{rid}", json={"name": "East Wing, Floor 2"}).json()["name"] == "East Wing, Floor 2"
    assert client.get(f"/maps/{rid}/pack").json()["name"] == "East Wing, Floor 2"
    client.patch(f"/maps/{rid}", json={"hidden": True})
    assert rid not in [m["id"] for m in client.get("/maps").json()["maps"]]
    assert rid in [m["id"] for m in client.get("/maps?include_hidden=true").json()["maps"]]
    client.patch(f"/maps/{rid}", json={"hidden": False})


def test_write_token(client, monkeypatch):
    rid = _completed_map("mobilemap3")
    monkeypatch.setenv("ACCESSMAP_API_TOKEN", "s3cret")
    assert client.patch(f"/maps/{rid}", json={"name": "A"}).status_code == 401
    assert client.patch(f"/maps/{rid}", json={"name": "A"}, headers={"x-accessmap-token": "nope"}).status_code == 401
    assert client.patch(f"/maps/{rid}", json={"name": "A"}, headers={"x-accessmap-token": "s3cret"}).status_code == 200
    assert client.patch(f"/maps/{rid}", json={"name": "B"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get(f"/maps/{rid}/pack").status_code == 200                # reads stay open


def _fake_groq(script):
    """MockTransport answering /models and replaying `script` (a list of assistant messages) for completions."""
    calls = []

    def handler(req: httpx.Request):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-large-v3"}, {"id": "llama-3.3-70b-versatile"}]})
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json={"choices": [{"message": script[min(len(calls), len(script)) - 1]}]})
    return handler, calls


def test_assistant_plans_real_route_with_tool(client, monkeypatch):
    import vision.assistant.groq as G
    rid = _completed_map("mobilemap4")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    G._models.update(at=0.0, ids=[])
    handler, calls = _fake_groq([
        {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "plan_route", "arguments": json.dumps({"goal": "graphics lab", "profile": "wheelchair"})}}]},
        {"content": "**Sorry** - there is no step-free way; please ask staff about a lift."},
    ])
    monkeypatch.setattr(G, "_client", lambda: httpx.Client(base_url=G.BASE, transport=httpx.MockTransport(handler)))
    assert client.get("/assistant/status").json() == {"available": True, "model": "llama-3.3-70b-versatile"}
    r = client.post("/assistant/chat", json={"map_id": rid, "profile": "wheelchair",
                                             "messages": [{"role": "user", "content": "How do I get to the graphics lab?"}]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["reply"] == "Sorry - there is no step-free way; please ask staff about a lift."   # markdown stripped
    assert d["tools_used"] == [{"tool": "plan_route", "args": {"goal": "graphics lab", "profile": "wheelchair"}, "ok": False}]
    sys_prompt = calls[0]["messages"][0]["content"]
    assert "CG Lab (lab, also called graphics lab)" in sys_prompt and "Wheelchair" in sys_prompt
    tool_result = json.loads(calls[1]["messages"][-1]["content"])
    assert tool_result["feasible"] is False and "only via stairs" in tool_result["reason"]
    assert calls[0]["model"] == "llama-3.3-70b-versatile" and calls[0]["tools"][0]["function"]["name"] == "plan_route"


def test_assistant_errors(client, monkeypatch):
    import vision.assistant.groq as G
    rid = _completed_map("mobilemap5")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert client.get("/assistant/status").json()["available"] is False
    msg = {"map_id": rid, "messages": [{"role": "user", "content": "hi"}]}
    assert client.post("/assistant/chat", json=msg).status_code == 503
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert client.post("/assistant/chat", json={**msg, "map_id": "nope"}).status_code == 404
    assert client.post("/assistant/chat", json={"map_id": rid, "messages": "hi"}).status_code == 400
    G._models.update(at=0.0, ids=[])

    def limited(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "llama-3.1-8b-instant"}]})
        return httpx.Response(429, json={})
    monkeypatch.setattr(G, "_client", lambda: httpx.Client(base_url=G.BASE, transport=httpx.MockTransport(limited)))
    assert client.post("/assistant/chat", json=msg).status_code == 429


def test_dotenv_loader(tmp_path, monkeypatch):
    from vision.config import load_dotenv
    f = tmp_path / ".env"
    f.write_text('# c\nA_ONE=1\nexport A_TWO="two words"\nA_THREE=keep\nbroken line\n')
    monkeypatch.delenv("A_ONE", raising=False)
    monkeypatch.delenv("A_TWO", raising=False)
    monkeypatch.setenv("A_THREE", "env wins")
    load_dotenv(f)
    import os
    assert os.environ["A_ONE"] == "1" and os.environ["A_TWO"] == "two words" and os.environ["A_THREE"] == "env wins"
    assert not math.isnan(1.0)
