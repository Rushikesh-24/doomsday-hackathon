"""Wayfinding assistant backed by Groq (OpenAI-compatible chat completions with tool calling).

The model never invents routes: it answers from the map's real places and calls `plan_route`, which runs
the same accessibility-aware router and directions as the rest of the platform. Replies are short and
speakable (they are read aloud to low-vision users). The API key stays on the server (GROQ_API_KEY).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)
BASE = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# used when GROQ_MODEL is unset or not available for this key (first match wins)
PREFERENCE = ["llama-3.3-70b", "llama-4-maverick", "llama-4-scout", "gpt-oss-120b", "qwen3", "kimi-k2",
              "llama-3.1-8b", "gpt-oss-20b"]
_models: dict = {"at": 0.0, "ids": []}
MAX_TURNS, MAX_CHARS, TOOL_ROUNDS = 16, 2000, 3


class AssistantError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def _key() -> str | None:
    k = os.environ.get("GROQ_API_KEY", "").strip()
    return k or None


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE, timeout=httpx.Timeout(30.0, connect=8.0),
                        headers={"Authorization": f"Bearer {_key()}"})


def _available_models() -> list[str]:
    if time.time() - _models["at"] < 3600 and _models["ids"]:
        return _models["ids"]
    with _client() as c:
        r = c.get("/models")
    if r.status_code == 401:
        raise AssistantError(503, "GROQ_API_KEY was rejected by Groq (401). Check the key in .env.")
    r.raise_for_status()
    ids = sorted(m["id"] for m in r.json().get("data", []) if m.get("active", True))
    _models.update(at=time.time(), ids=ids)
    return ids


def pick_model() -> str:
    want = os.environ.get("GROQ_MODEL", "").strip()
    ids = _available_models()
    if want and want in ids:
        return want
    chat = [i for i in ids if not re.search(r"whisper|tts|guard|embed|vision|playai|orpheus|distil", i)]
    for p in PREFERENCE:
        hit = next((i for i in chat if p in i), None)
        if hit:
            return hit
    if not chat:
        raise AssistantError(503, "no chat model is available for this Groq key")
    return chat[0]


def status() -> dict:
    if not _key():
        return {"available": False, "reason": "GROQ_API_KEY is not set on the server (.env)"}
    try:
        return {"available": True, "model": pick_model()}
    except AssistantError as e:
        return {"available": False, "reason": str(e)}
    except httpx.HTTPError as e:
        return {"available": False, "reason": f"cannot reach Groq: {e.__class__.__name__}"}


SYSTEM = """You are AccessMap, a calm, friendly indoor wayfinding guide for a campus building. You help people \
with different needs: wheelchair users, people with low vision, and anyone walking.
Rules:
- Use ONLY the places and facilities listed below. If something is not on the map, say so plainly and suggest \
asking staff; never invent rooms, distances or directions.
- For any "how do I get to / where is / take me to" request, call plan_route and describe the result. \
Do not describe a route without calling the tool.
- Keep replies short (1-4 sentences), concrete and easy to read aloud. No markdown, no emojis, no lists of symbols.
- Respect the traveller's profile ({profile_label}). If no step-free route exists for a wheelchair user, say so \
and suggest a lift, a ramp or asking staff.
- Safety first: for stairs, crowds or obstacles give practical, respectful advice. In an emergency tell them to \
contact campus security or local emergency services.
Map: {map_name}. Units: {units}. Default start: {default_start}.
Places: {places}
Facilities detected in 3D: {facilities}
{route_context}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "plan_route",
        "description": "Plan an accessible walking route on this map and get turn-by-turn directions.",
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "Destination: a place name/alias from the list, or a facility "
                                                      "type such as 'elevator', 'ramp', 'toilet_sign', 'entrance_exit'."},
            "start": {"type": "string", "description": "Optional start place name. Omit to use the traveller's start."},
            "profile": {"type": "string", "enum": ["standard", "wheelchair", "low_vision"],
                        "description": "Omit to use the traveller's profile."}},
            "required": ["goal"]}}},
]


def _graph(job: Path) -> dict:
    from vision.navigation.graph import build
    from vision.navigation.places import attach_places
    return attach_places(build(job), job)


def _plan(job: Path, g: dict, args: dict, profile: str, start: str | None) -> dict:
    from vision.navigation.graph import route
    from vision.navigation.places import directions, resolve
    try:
        st = resolve(g, args.get("start") or start or "start")
        rt = directions(g, route(g, st, resolve(g, args["goal"]), args.get("profile") or profile))
    except (KeyError, ValueError) as e:
        return {"ok": False, "error": str(e).strip("'\"")}
    if not rt["feasible"]:
        return {"ok": False, "feasible": False, "reason": rt.get("reason"), "profile": rt["profile"], "route": rt}
    return {"ok": True, "from": rt["start_name"], "to": rt["goal_name"], "length": rt["length"], "units": rt["units"],
            "warnings": rt.get("warnings"), "directions": [s["text"] + (" " + " ".join(s["notes"]) if s.get("notes") else "")
                                                         for s in rt["directions"]], "route": rt}


def chat(job: Path, map_name: str, messages: list[dict], profile: str = "standard", start: str | None = None,
         current_route: dict | None = None) -> dict:
    if not _key():
        raise AssistantError(503, "The assistant is not configured: add GROQ_API_KEY to the server's .env.")
    from vision.navigation.graph import PROFILES
    if profile not in PROFILES:
        raise AssistantError(400, f"profile must be one of {list(PROFILES)}")
    msgs = [m for m in messages if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    msgs = [{"role": m["role"], "content": m["content"][:MAX_CHARS]} for m in msgs][-MAX_TURNS:]
    if not msgs or msgs[-1]["role"] != "user" or not msgs[-1]["content"].strip():
        raise AssistantError(400, "the last message must be a non-empty user message")
    g = _graph(job)
    places = "; ".join(
        f"{p['name']} ({p['kind']}" + (f", also called {', '.join(p['aliases'])}" if p.get("aliases") else "")
        + (", step-free" if (p.get("access") or {}).get("step_free") else "")
        + (f", note: {p['description']}" if p.get("description") else "") + ")" for p in g.get("places", [])) or "none tagged"
    fac = {}
    for n in g["nodes"]:
        if n["kind"] == "facility":
            fac[n["class_name"]] = fac.get(n["class_name"], 0) + 1
    default_start = next((p["name"] for p in g.get("places", []) if p.get("is_start")), "the start of the scanned walk")
    route_ctx = ""
    if current_route and current_route.get("directions"):
        steps = " | ".join(s.get("text", "") for s in current_route["directions"])[:1500]
        route_ctx = f"The traveller is currently following this route: {steps}"
    system = SYSTEM.format(profile_label=PROFILES[profile]["label"], map_name=map_name, units=
                           "metres" if g["units"] == "m" else "relative units (no metric scale)",
                           default_start=default_start, places=places,
                           facilities=", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in sorted(fac.items())) or "none",
                           route_context=route_ctx)
    convo = [{"role": "system", "content": system}, *msgs]
    model = pick_model()
    planned, used = None, []
    with _client() as c:
        for _ in range(TOOL_ROUNDS + 1):
            body = {"model": model, "messages": convo, "temperature": 0.3, "max_tokens": 400,
                    "tools": TOOLS, "tool_choice": "auto"}
            r = c.post("/chat/completions", json=body)
            if r.status_code == 429:
                raise AssistantError(429, "The assistant is busy (Groq rate limit). Try again in a few seconds.")
            if r.status_code == 401:
                raise AssistantError(503, "GROQ_API_KEY was rejected by Groq (401). Check the key in .env.")
            if r.status_code >= 400:
                log.warning("groq error %s: %s", r.status_code, r.text[:500])
                raise AssistantError(502, f"Groq returned an error ({r.status_code}).")
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                text = re.sub(r"[*_#`]+", "", text)            # speakable plain text
                return {"reply": text, "model": model, "tools_used": used,
                        "route": planned["route"] if planned and planned.get("route") else None}
            convo.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "plan_route" and isinstance(args.get("goal"), str):
                    res = _plan(job, g, args, profile, start)
                    planned = res if res.get("route") else planned
                    used.append({"tool": name, "args": args, "ok": res.get("ok", False)})
                    payload = {k: v for k, v in res.items() if k != "route"}
                else:
                    payload = {"ok": False, "error": f"unknown tool or bad arguments: {name}"}
                convo.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(payload)[:4000]})
    return {"reply": "Sorry, I could not work that out. Please try asking in a different way.", "model": model,
            "tools_used": used, "route": planned["route"] if planned and planned.get("route") else None}
