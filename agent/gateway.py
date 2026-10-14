"""Solo VLearn control plane. Returns synchronous, pure Decisions for ROUTE, ADMIT, AUTHORIZE and BUDGET. Command/Decision schemas follow the shipped arena contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, runtime_checkable

# kit.mcp.types is a collaborator's file (workspace hard rule 2: import it,
# degrade gracefully). It is present as of this writing and is core, stable
# infrastructure (CONTRACTS.md 3.1) — but this module must still not fail to
# IMPORT if a concurrent edit ever breaks it transiently. When it is
# unavailable, `Decision.call` type-checking is skipped (not enforced), and
# `Gateway.decide` falls back to a minimal local dict-shaped stand-in so the
# rest of this file — everything that does not need a *real* ToolCall — still
# runs.
try:
    from kit.mcp.types import ToolCall
    _TOOLCALL_AVAILABLE = True
except ImportError:  # pragma: no cover - collaborator file
    ToolCall = Any  # type: ignore[assignment, misc]
    _TOOLCALL_AVAILABLE = False

# kit.loop.agent is also a collaborator's file, used only by this module's
# own __main__ demo (to build real Commands the same way the arena's trusted
# canonicaliser would) — never by decide() itself, which never touches the
# loop. Degraded the same way.
try:
    from kit.loop.agent import canonicalise_action as _canonicalise_action
except ImportError:  # pragma: no cover - collaborator file
    _canonicalise_action = None

from agent.telemetry import RecordingGatewayContext, Telemetry

__all__ = [
    "COMMAND_KINDS",
    "DECISION_VERDICTS",
    "Command",
    "Decision",
    "GatewayContext",
    "Gateway",
]

# CONTRACTS.md 4.1: `Command.kind` — "mcp" | "a2a" | "discover". An "answer"
# action is NEVER a Command (kit/loop/agent.py's own module docstring: "an
# answer is not a tool call routed to a server, so it never becomes a
# Command at all") — it is handled entirely by the loop/arena and never
# reaches `Gateway.decide`.
COMMAND_KINDS: frozenset[str] = frozenset({"mcp", "a2a", "discover"})

# CONTRACTS.md 4.1: `Decision.verdict` — the closed three-member set.
DECISION_VERDICTS: frozenset[str] = frozenset({"forward", "deny", "rewrite"})


@dataclass(frozen=True, slots=True)
class Command:
    """CONTRACTS.md 4.1, field for field — "canonicalised by the arena
    BEFORE the student sees it". You never build one of these from your own
    agent's raw text; the arena's canonicaliser (kit/loop/agent.py's
    `canonicalise_action`, run inside the trusted envelope) already did that
    work and minted `cmd_id` by the time `decide()` sees it. The
    `from_action_dict` classmethod below exists only so this file's own demo
    (and your local tests, if you write any) can build a realistic `Command`
    without duplicating the arena's canonicalisation logic."""

    cmd_id: str
    kind: str  # "mcp" | "a2a" | "discover" — see COMMAND_KINDS
    raw: str
    server: str
    tool: str
    args: dict
    fields: tuple[str, ...]
    headers: dict
    lease_id: str | None
    call_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cmd_id, str) or not self.cmd_id:
            raise ValueError(f"Command.cmd_id must be a non-empty str, got {self.cmd_id!r}")
        if self.kind not in COMMAND_KINDS:
            raise ValueError(f"Command.kind must be one of {sorted(COMMAND_KINDS)}, got {self.kind!r}")
        if not isinstance(self.server, str) or not self.server:
            raise ValueError(f"Command.server must be a non-empty str, got {self.server!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise ValueError(f"Command.tool must be a non-empty str, got {self.tool!r}")
        if not isinstance(self.args, dict):
            raise ValueError(f"Command.args must be a dict, got {type(self.args).__name__}")
        if not isinstance(self.headers, dict):
            raise ValueError(f"Command.headers must be a dict, got {type(self.headers).__name__}")
        if (
            not isinstance(self.call_index, int)
            or isinstance(self.call_index, bool)
            or self.call_index < 0
        ):
            raise ValueError(f"Command.call_index must be a non-negative int, got {self.call_index!r}")

    @classmethod
    def from_action_dict(cls, action: Mapping[str, Any], *, cmd_id: str) -> "Command":
        """Build a `Command` from the dict shape `kit.loop.agent.canonicalise_action`
        returns (`kind, raw, server, tool, args, fields, headers, lease_id,
        call_index` — everything except the arena-minted `cmd_id`, supplied
        here as a keyword). Raises `ValueError` if `action["kind"] ==
        "answer"` — an answer is never a Command (see the module docstring).
        This is a convenience for tests/demos, not something the real arena
        calls: the trusted envelope mints `cmd_id` itself and constructs the
        real `Command` on its own side of the boundary."""
        kind = action.get("kind")
        if kind == "answer":
            raise ValueError(
                "an 'answer' action never becomes a Command (kit/loop/agent.py: "
                "\"an answer is not a tool call routed to a server\") — do not "
                "route it through Gateway.decide at all"
            )
        return cls(
            cmd_id=cmd_id,
            kind=kind,
            raw=action["raw"],
            server=action["server"],
            tool=action["tool"],
            args=dict(action.get("args", {})),
            fields=tuple(action.get("fields", ())),
            headers=dict(action.get("headers", {})),
            lease_id=action.get("lease_id"),
            call_index=action.get("call_index", 0),
        )

    def to_dict(self) -> dict:
        return {
            "cmd_id": self.cmd_id,
            "kind": self.kind,
            "raw": self.raw,
            "server": self.server,
            "tool": self.tool,
            "args": dict(self.args),
            "fields": list(self.fields),
            "headers": dict(self.headers),
            "lease_id": self.lease_id,
            "call_index": self.call_index,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """CONTRACTS.md 4.1, field for field.

    Validated strictly (`__post_init__`) because a *structurally* invalid
    `Decision` is charged exactly like a raised exception — CONTRACTS.md
    4.1's charging table: "malformed Decision (schema-invalid) -> 2 cr
    penalty, command denied." Failing loudly HERE, in your own process
    during development, is strictly better than discovering it live in a
    duel as an unexplained penalty.

    `verdict == "deny"` requires a non-empty `reason` (CONTRACTS.md 4.1:
    "required when verdict == 'deny'; shown in the combat log") and
    forbids `call` — a real denial has nothing left to carry out.
    `verdict` in `("forward", "rewrite")` requires `call` to be set — the
    arena executes exactly that `ToolCall`, nothing else, per the trusted
    envelope's whole point (see the module docstring)."""

    verdict: str  # "forward" | "deny" | "rewrite" — see DECISION_VERDICTS
    reason: str | None = None
    call: "ToolCall | None" = None
    quarantine: bool = False
    note: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in DECISION_VERDICTS:
            raise ValueError(
                f"Decision.verdict must be one of {sorted(DECISION_VERDICTS)}, got {self.verdict!r}"
            )
        if self.verdict == "deny":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("Decision.verdict=='deny' requires a non-empty 'reason'")
            if self.call is not None:
                raise ValueError("Decision.verdict=='deny' must not carry a 'call' — there is nothing to run")
        else:  # forward | rewrite
            if self.call is None:
                raise ValueError(f"Decision.verdict=={self.verdict!r} requires 'call' to be set")
            if _TOOLCALL_AVAILABLE and not isinstance(self.call, ToolCall):
                raise ValueError(
                    f"Decision.call must be a kit.mcp.types.ToolCall instance, got {type(self.call).__name__}"
                )
        if not isinstance(self.quarantine, bool):
            raise ValueError(f"Decision.quarantine must be a bool, got {self.quarantine!r}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValueError(f"Decision.note must be a str or None, got {self.note!r}")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "call": self.call.to_dict() if self.call is not None and hasattr(self.call, "to_dict") else self.call,
            "quarantine": self.quarantine,
            "note": self.note,
        }


@runtime_checkable
class GatewayContext(Protocol):
    """CONTRACTS.md 4.2 — "read-only, arena-provided". Note what this is
    NOT: unlike `Command`/`Decision` above, CONTRACTS.md writes this as a
    plain `class`, not a `@dataclass(frozen=True, ...)`. That is a real
    signal, not an inconsistency — a `GatewayContext` is a LIVE VIEW the
    arena owns and updates as the duel actually progresses (credits get
    spent, rounds advance, leases mint and expire), not a frozen snapshot
    you receive once. This file therefore models it as a structural
    `Protocol`: the arena hands your `Gateway.__init__` some object shaped
    like this (see `agent/telemetry.py`'s `RecordingGatewayContext` for a
    concrete example your own tests/demos can build), and you read
    `self.ctx.<field>` fresh every time `decide()` runs — NEVER cache a
    context field into a local variable across calls; if you need memory
    that persists across the duel (a cache, a budget tracker, drift notes),
    keep it on YOUR OWN `Gateway` instance attributes instead (CONTRACTS.md
    4.3: "Gateway is instantiated once per duel... persists across the 10
    rounds").

    There is deliberately no `execute` method here — see the module
    docstring's "WHY THERE IS NO execute() METHOD" section. `emit` is the
    ONE thing you can do that has any side effect at all, and it writes to
    `own_telemetry` only: your own side, never scored, never shown to the
    opponent (`agent/telemetry.py` wraps it)."""

    act: str  # "learner:sv-0417" — WHOM YOU SERVE. Authority derives from this.
    sub: str  # "agent:vlearn-tutor" — what you are. NOT authority.
    scopes: frozenset[str]  # {"wiki.read"}; writes need "wiki.write:progress" etc.
    credits: int  # remaining this duel
    round: int
    call_index: int
    leases: tuple[str, ...]  # live lease ids, arena-tracked
    history: tuple[Mapping[str, Any], ...]  # YOUR OWN prior (Command, Decision, outcome) triples this duel

    def emit(self, name: str, **payload: Any) -> None: ...


class Gateway:
    """One gateway per duel. Uses trusted context, admitted peer cards, provenance and per-round spend; never executes a tool."""

    def __init__(self, ctx: GatewayContext) -> None:
        self.ctx = ctx
        self._telemetry = Telemetry(ctx)

        # --- per-duel memory, unused by the naive starter below ---------
        # A cache of anchor -> body-ish data you have already paid for this
        # duel (agent/strategy.py's ResultCache is a ready-made version of
        # this). Populating it needs the *result* of a call, which decide()
        # never sees (it only sees the outgoing Command) — you would fill
        # this from whatever the arena hands back to your agent loop AFTER
        # a call executes, then consult it here on the NEXT decide() call
        # for the same anchor.
        self._seen_anchors: dict[str, Any] = {}
        # Credits you have personally authorised so far this duel — your
        # own running total, independent of (and a cross-check against)
        # `ctx.credits`, which the arena maintains authoritatively.
        self._credits_authorised: int = 0
        # Command ids you have already denied, in case a later job wants to
        # know "have I already said no to this once".
        self._denied_cmd_ids: set[str] = set()

    def note_card(self, server: str, card: dict) -> None:
        # Only the trusted host may feed registry admission results here.
        if not hasattr(self, '_cards'):
            self._cards = {}
        self._cards[server] = dict(card)

    def note_provenance(self, anchor: str, etag: str) -> None:
        self._seen_anchors[anchor] = etag

    def decide(self, cmd: Command) -> Decision:
        """ROUTE, ADMIT, AUTHORIZE, BUDGET; synchronous and without I/O."""
        from agent.strategy import cheap_mask, successor_of, pick_replica
        from agent.guardrails import scan_for_injected_instructions
        from kit.mcp.specs import TOOL_SPECS
        from kit.mcp.a2a import verify_card, verify_delegation, admit_skill

        headers = {str(k).lower(): v for k, v in cmd.headers.items()}
        # ROUTE: a body route or contradictory header is untrusted input.
        if any(cmd.args.get(k) for k in ('route', '_route', 'replica')):
            return self.deny(cmd, 'replica routing must be carried by the header')
        if headers.get('x-server-fingerprint') == 'unvouched':
            return self.deny(cmd, 'server identity is not vouched for by registry')
        if headers.get('x-card-signature') == 'invalid':
            return self.deny(cmd, 'invalid peer card signature')
        if cmd.args.get('peer_unverified'):
            return self.deny(cmd, 'peer result requires independent corroboration')
        if headers.get('x-mcp-body-route'):
            return self.deny(cmd, 'body routing is not authorized')
        for key in ('note', 'body', 'instructions', 'peer_answer'):
            if scan_for_injected_instructions(str(cmd.args.get(key, ''))).suspicious:
                return self.deny(cmd, 'retrieved content contains an injected instruction')

        server, tool = successor_of(cmd.server, cmd.tool) or (cmd.server, cmd.tool)
        spec = TOOL_SPECS.get((server, tool))
        if spec is None:
            return self.deny(cmd, 'unknown server/tool')
        if not spec.is_write and 'wiki.read' not in self.ctx.scopes:
            return self.deny(cmd, 'required read scope was not granted')
        if headers.get('mcp-replica') not in (None, 'w', 'c'):
            return self.deny(cmd, 'invalid replica header')
        headers.setdefault('mcp-replica', pick_replica(path_id=None, known_drifting=False).replica)

        # ADMIT: fresh leases and no blind retries after a definitive failure.
        if spec.needs_lease and (not cmd.lease_id or cmd.lease_id not in self.ctx.leases):
            return self.deny(cmd, 'get_frame requires a live query lease')
        for entry in self.ctx.history:
            if not isinstance(entry, Mapping):
                continue
            previous = entry.get('command', {})
            result = entry.get('outcome', {})
            if not isinstance(previous, Mapping) or not isinstance(result, Mapping):
                continue
            if (previous.get('server'), previous.get('tool'), previous.get('args')) == (server, tool, cmd.args):
                error = result.get('error') or {}
                if isinstance(error, Mapping) and error.get('code') in ('conflict', 'bad_request', 'unauthorized', 'not_found'):
                    if previous.get('headers', {}) == cmd.headers:
                        return self.deny(cmd, 'failed call must be corrected before retry')

        # AUTHORIZE: identity (sub) never grants authority over another act.
        act = str(self.ctx.act).casefold()
        for key in ('learner', 'learner_id', 'for_learner', 'target', 'subject'):
            target = cmd.args.get(key)
            if target and (spec.is_write or cmd.kind == 'a2a' or server == 'roster'):
                if str(target).casefold() != act:
                    return self.deny(cmd, 'target learner differs from ctx.act')
        if spec.is_write:
            if f'wiki.write:{server}' not in self.ctx.scopes:
                return self.deny(cmd, 'required write scope was not granted')
            anchor = str(cmd.args.get('anchor') or cmd.args.get('learner') or '')
            etag = self._seen_anchors.get(anchor)
            if not etag or headers.get('if-match') != etag:
                return self.deny(cmd, 'write requires a matching provenance etag')
            key = headers.get('idempotency-key')
            if not isinstance(key, str) or not key:
                return self.deny(cmd, 'write requires an idempotency key')
            if not hasattr(self, '_write_keys'):
                self._write_keys = set()
            if key in self._write_keys:
                return self.deny(cmd, 'write key already authorized this duel')

        if cmd.kind == 'a2a':
            aud = headers.get('aud')
            if aud not in (server, f'a2a:{server}', f'mcp:{server}'):
                return self.deny(cmd, 'delegation audience does not match peer')
            card = getattr(self, '_cards', {}).get(server)
            if not card:
                return self.deny(cmd, 'peer card has not been admitted')
            if 'signature' in card:
                if not verify_card(card).admitted or not admit_skill(card, tool).admitted:
                    return self.deny(cmd, 'peer signature or declared skill is invalid')
            elif not card.get('verified') or tool not in card.get('skills', ()):
                return self.deny(cmd, 'peer is not verified for this skill')
            token = headers.get('delegation') or cmd.args.get('delegation')
            if token is not None:
                seen = getattr(self, '_token_ids', set())
                admission = verify_delegation(token, aud=f'a2a:{server}', call_index=cmd.call_index,
                                              expected_act=self.ctx.act, seen_token_ids=seen)
                if not admission.admitted:
                    return self.deny(cmd, 'delegation signature, act, audience, expiry or replay check failed')
            elif not card.get('verified'):
                return self.deny(cmd, 'signed delegation is required')

        # BUDGET: preserve useful fields; wildcard catalogs never go through.
        fields = tuple(cmd.fields)
        if fields and fields != ('*',) and any(f not in spec.all_fields for f in fields):
            return self.deny(cmd, 'field mask contains an unknown field')
        if not fields or '*' in fields:
            desired = {'list_servers': ('name',), 'list_terms': ('term', 'anchor')}.get(tool, spec.default_fields)
            fields = cheap_mask(server, tool, tuple(f for f in desired if f in spec.all_fields))
        args = dict(cmd.args)
        if 'catalog' in args and len(str(args['catalog'])) > 1024:
            return self.deny(cmd, 'inflated catalog exceeds bounded discovery request')
        # Bound result cardinality for query tools without truncating fetched bodies.
        if tool in ('query', 'list_servers', 'list_terms'):
            limit = args.get('limit', 3)
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                return self.deny(cmd, 'limit must be a positive integer')
            args['limit'] = min(limit, 3)
        rows = args.get('limit', 1) if spec.row_weight else 1
        if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
            return self.deny(cmd, 'row count must be a positive integer')
        price = spec.base + sum(spec.field_weight.get(f, 0) for f in fields) + spec.row_weight * rows
        if price > self.ctx.credits:
            minimal = tuple(spec.default_fields)
            cheaper = spec.base + sum(spec.field_weight.get(f, 0) for f in minimal) + spec.row_weight * rows
            if cheaper <= self.ctx.credits and cheaper < price:
                fields, price = minimal, cheaper
            else:
                return self.deny(cmd, 'remaining credits cannot fund the requested mask')
        if not hasattr(self, '_round_spend'):
            self._round_spend = {}
        rnd = self.ctx.round
        allowance = max(8, self.ctx.credits // max(1, 11 - rnd))
        if self._round_spend.get(rnd, 0) + price > allowance:
            return self.deny(cmd, 'round allowance exhausted; preserve credits for later rounds')
        routed = replace(cmd, server=server, tool=tool, args=args, fields=fields, headers=headers)
        decision = Decision(verdict='rewrite' if routed != cmd else 'forward', call=self._to_tool_call(routed))
        self._round_spend[rnd] = self._round_spend.get(rnd, 0) + price
        self._credits_authorised += price
        if spec.is_write:
            self._write_keys.add(headers['idempotency-key'])
        if cmd.kind == 'a2a' and token is not None:
            if not hasattr(self, '_token_ids'):
                self._token_ids = set()
            self._token_ids.add(token.get('token_id') if isinstance(token, Mapping) else token.token_id)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def deny(self, cmd: Command, reason: str) -> Decision:
        """Record and return a valid, zero-credit denial with a concrete reason."""
        self._denied_cmd_ids.add(cmd.cmd_id)
        decision = Decision(verdict="deny", reason=reason)
        self._telemetry.decision_made(cmd, decision)
        return decision

    def _to_tool_call(self, cmd: Command) -> "ToolCall":
        """`Command` -> the `ToolCall` (CONTRACTS.md 3.1) the arena will
        actually execute on a `forward`/`rewrite` verdict. When
        `kit.mcp.types` is unavailable (see the module-level import guard),
        falls back to a plain dict carrying the identical fields — `Decision`
        accepts it either way (the `ToolCall` isinstance check inside
        `Decision.__post_init__` only runs when the real class loaded)."""
        fields = {
            "server": cmd.server,
            "tool": cmd.tool,
            "args": dict(cmd.args),
            "fields": cmd.fields,
            "headers": dict(cmd.headers),
            "lease_id": cmd.lease_id,
            "call_index": cmd.call_index,
        }
        if _TOOLCALL_AVAILABLE:
            return ToolCall(**fields)
        return fields  # type: ignore[return-value]


if __name__ == "__main__":
    print("=== agent.gateway: Command / Decision validation ===\n")

    good_cmd = Command(
        cmd_id="cmd:0000",
        kind="mcp",
        raw="MCP slides.get_frame anchor=Frame:3f2a9c11/w/041 fields=title,body lease=lse_7f21",
        server="slides",
        tool="get_frame",
        args={"anchor": "Frame:3f2a9c11/w/041"},
        fields=("body", "title"),
        headers={},
        lease_id="lse_7f21",
        call_index=0,
    )
    print(f"  Command constructed: {good_cmd}")
    assert good_cmd.kind == "mcp"

    print("\n  Rejection demo (each must raise ValueError):")

    def _expect_value_error(label: str, fn) -> None:
        try:
            fn()
        except ValueError as exc:
            print(f"    [{label:38}] -> ValueError: {exc}")
        else:
            raise AssertionError(f"expected ValueError for case {label!r}")

    _expect_value_error("Command.kind == 'answer'", lambda: Command(
        cmd_id="cmd:0001", kind="answer", raw="x", server="slides", tool="get_frame",
        args={}, fields=(), headers={}, lease_id=None, call_index=0,
    ))
    _expect_value_error("Decision verdict='deny' with no reason", lambda: Decision(verdict="deny"))
    _expect_value_error(
        "Decision verdict='forward' with no call", lambda: Decision(verdict="forward")
    )
    _expect_value_error(
        "Decision verdict='deny' carrying a call",
        lambda: Decision(verdict="deny", reason="nope", call={"server": "x", "tool": "y"}),
    )
    _expect_value_error("Decision verdict='?' unknown", lambda: Decision(verdict="???"))

    print("\n=== Command.from_action_dict — real canonicaliser integration ===\n")
    if _canonicalise_action is None:
        print("  kit.loop.agent not importable yet — skipping the live canonicaliser demo")
        demo_commands: list[Command] = [good_cmd]
    else:
        raw_actions = [
            "MCP registry.provenance anchor=Frame:3f2a9c11/w/041 fields=etag",
            'MCP slides.query q="streamable http replaces http+sse" fields=title,body',
            "A2A curriculum-analyst.which_days_cover concept=Concept:streamable-http fields=anchor,course_day,track",
            "DISCOVER registry.list_servers fields=name",
        ]
        demo_commands = []
        for i, raw in enumerate(raw_actions):
            action = _canonicalise_action(raw, call_index=i)
            cmd = Command.from_action_dict(action, cmd_id=f"cmd:{i:04d}")
            print(f"  {raw!r}\n    -> {cmd.kind}: {cmd.server}.{cmd.tool} fields={cmd.fields}")
            demo_commands.append(cmd)
        assert {c.kind for c in demo_commands} == {"mcp", "a2a", "discover"}

        answer_action = _canonicalise_action(
            'ANSWER {"text": "day 26, track P2T2"}', call_index=None
        )
        try:
            Command.from_action_dict(answer_action, cmd_id="cmd:9999")
        except ValueError as exc:
            print(f"\n  an 'answer' action correctly refuses to become a Command: {exc}")
        else:
            raise AssertionError("expected ValueError for an 'answer' action")

    print("\n=== Gateway.decide — the naive starter forwards everything ===\n")
    ctx = RecordingGatewayContext(
        act="learner:sv-0401",
        sub="agent:demo-team",
        scopes=frozenset({"wiki.read"}),
        credits=100,
        round=1,
        call_index=0,
        leases=(),
        history=(),
    )
    assert isinstance(ctx, GatewayContext), "RecordingGatewayContext must structurally satisfy GatewayContext"
    gw = Gateway(ctx)
    for cmd in demo_commands:
        decision = gw.decide(cmd)
        print(f"  decide({cmd.server}.{cmd.tool}) -> verdict={decision.verdict!r} quarantine={decision.quarantine}")
        assert decision.verdict == "forward"
        assert decision.call is not None
        call_dict = decision.call.to_dict() if hasattr(decision.call, "to_dict") else decision.call
        assert call_dict["server"] == cmd.server
        assert call_dict["tool"] == cmd.tool
        assert tuple(call_dict["fields"]) == cmd.fields

    print(f"\n=== Gateway.deny — the unused-by-default free-abstention path ===\n")
    denial = gw.deny(demo_commands[0], reason="demo: withholding pending a fresher registry.provenance read")
    print(f"  gw.deny(...) -> verdict={denial.verdict!r} reason={denial.reason!r} call={denial.call!r}")
    assert denial.verdict == "deny"
    assert denial.call is None
    assert demo_commands[0].cmd_id in gw._denied_cmd_ids

    print(f"\n=== own_telemetry — recorded on YOUR side only, never shown to the opponent ===\n")
    print(f"  {len(ctx.events)} events recorded on this ctx this run:")
    for ev in ctx.events:
        print(f"    {ev['name']}: {sorted(ev['payload'].keys())}")
    assert len(ctx.events) >= len(demo_commands) * 2 + 1  # decision_seen + decision_made per call, plus the deny

    print("\nAll agent/gateway.py demos passed.")
