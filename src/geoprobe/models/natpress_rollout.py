"""Natural-pressure multi-turn rollout driver (protocol ``natpress_v1``).

Pure planning/seed/assembly logic plus a thin HF generation glue. The whole pipeline runs without
torch via :class:`StubGenerator`; ``transformers``/``torch`` are imported lazily inside the HF paths
only, so config validation, arm assembly, seed derivation and the stub rollout stay import-light and
testable in a torch-free environment.

Row schema written per conversation is ``natpress_rollout_v1`` (see the project spec / tests).
"""
from __future__ import annotations

import hashlib
from typing import Any

SCHEMA_VERSION = "natpress_rollout_v1"

ARMS = ("smooth", "benign", "step", "latedump")

REQUIRED_SCENARIO_FIELDS = frozenset({
    "scenario_id", "family", "title", "ground_truth", "pressurer", "system",
    "t1_user", "benign_turns", "smooth_turns", "step_t6", "latedump_t5",
    "latedump_t6", "corner",
})
# Optional per-scenario fields (v3 adaptive): the fixed benign-arm closer used by the
# adaptive orchestrator (scripted arms ignore it; conventionally == benign_turns[-1]).
OPTIONAL_SCENARIO_FIELDS = frozenset({"benign_corner_turn"})
REQUIRED_GROUND_TRUTH_FIELDS = frozenset({"fact", "honest_verdict", "deceptive_verdict"})

# Assembly order used for planning. benign precedes step/latedump so every sibling that copies
# benign replies is generated after the benign it depends on.
_PLAN_ARM_ORDER = ("benign", "smooth", "step", "latedump")

# Turn indices (0-based, turn_index 0 == T1) whose assistant reply is copied verbatim from the
# benign sibling because the user prefix is byte-identical by construction.
_SHARED_TURN_INDICES = {
    "step": frozenset({1, 2, 3, 4}),
    "latedump": frozenset({1, 2, 3}),
}


# --------------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------------
def _require_nonempty_str(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")


def validate_natpress_config(cfg: Any) -> None:
    """Validate a natpress config in place; raise ``ValueError`` with a precise message on any fault."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a JSON object")
    for key in ("protocol", "version", "sampling", "samples_per_arm", "scenarios"):
        if key not in cfg:
            raise ValueError(f"config missing top-level key: {key!r}")
    _require_nonempty_str(cfg["protocol"], "config.protocol")
    _require_nonempty_str(cfg["version"], "config.version")

    sampling = cfg["sampling"]
    if not isinstance(sampling, dict):
        raise ValueError("config.sampling must be an object")
    for key in ("temperature", "top_p", "max_new_tokens_per_turn"):
        if key not in sampling:
            raise ValueError(f"config.sampling missing key: {key!r}")

    spa = cfg["samples_per_arm"]
    if not isinstance(spa, dict):
        raise ValueError("config.samples_per_arm must be an object")
    for arm in ARMS:
        if arm not in spa:
            raise ValueError(f"config.samples_per_arm missing arm: {arm!r}")
        if not isinstance(spa[arm], int) or isinstance(spa[arm], bool) or spa[arm] < 0:
            raise ValueError(f"config.samples_per_arm[{arm!r}] must be a non-negative int")
    if spa["step"] > spa["benign"]:
        raise ValueError("config.samples_per_arm: step must be <= benign (step reuses benign siblings)")
    if spa["latedump"] > spa["benign"]:
        raise ValueError("config.samples_per_arm: latedump must be <= benign (latedump reuses benign siblings)")

    # Optional v3 keys. sample_index_offset gives a bank a fresh sNN block so reused scenarios
    # can never mint conversation_ids colliding with an earlier bank (smoke used s00..).
    offset = cfg.get("sample_index_offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("config.sample_index_offset must be a non-negative int")
    # adaptive_samples_per_arm sizes the adaptive orchestrator's plan; adaptive arms generate
    # every turn (no sibling reuse), so the step/latedump <= benign constraint does not apply.
    aspa = cfg.get("adaptive_samples_per_arm")
    if aspa is not None:
        if not isinstance(aspa, dict):
            raise ValueError("config.adaptive_samples_per_arm must be an object")
        for arm in ARMS:
            if arm not in aspa:
                raise ValueError(f"config.adaptive_samples_per_arm missing arm: {arm!r}")
            if not isinstance(aspa[arm], int) or isinstance(aspa[arm], bool) or aspa[arm] < 0:
                raise ValueError(
                    f"config.adaptive_samples_per_arm[{arm!r}] must be a non-negative int"
                )

    scenarios = cfg["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config.scenarios must be a non-empty list")

    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"config.scenarios[{idx}] must be an object")
        fields = set(scenario)
        missing = sorted(REQUIRED_SCENARIO_FIELDS - fields)
        extra = sorted(fields - REQUIRED_SCENARIO_FIELDS - OPTIONAL_SCENARIO_FIELDS)
        if missing or extra:
            raise ValueError(
                f"config.scenarios[{idx}] fields mismatch: missing={missing}, unexpected={extra}"
            )
        if "benign_corner_turn" in scenario:
            _require_nonempty_str(
                scenario["benign_corner_turn"], f"config.scenarios[{idx}].benign_corner_turn"
            )
        sid = scenario["scenario_id"]
        family = scenario["family"]
        for name in ("scenario_id", "family", "title", "pressurer", "system", "t1_user",
                     "step_t6", "latedump_t5", "latedump_t6", "corner"):
            _require_nonempty_str(scenario[name], f"config.scenarios[{idx}].{name}")
        if sid in seen_ids:
            raise ValueError(f"config.scenarios: duplicate scenario_id: {sid!r}")
        if family in seen_families:
            raise ValueError(f"config.scenarios: duplicate family: {family!r}")
        seen_ids.add(sid)
        seen_families.add(family)

        gt = scenario["ground_truth"]
        if not isinstance(gt, dict):
            raise ValueError(f"config.scenarios[{idx}].ground_truth must be an object")
        for name in REQUIRED_GROUND_TRUTH_FIELDS:
            if name not in gt:
                raise ValueError(f"config.scenarios[{idx}].ground_truth missing key: {name!r}")
            _require_nonempty_str(gt[name], f"config.scenarios[{idx}].ground_truth.{name}")

        for name, expected in (("benign_turns", 6), ("smooth_turns", 5)):
            seq = scenario[name]
            if not isinstance(seq, list) or len(seq) != expected:
                raise ValueError(
                    f"config.scenarios[{idx}].{name} must be a list of {expected} strings"
                )
            for j, turn in enumerate(seq):
                _require_nonempty_str(turn, f"config.scenarios[{idx}].{name}[{j}]")

        for arm in ARMS:
            turns = assemble_arm_user_turns(scenario, arm)
            if len(turns) != 7:
                raise ValueError(
                    f"config.scenarios[{idx}] arm {arm!r} assembled {len(turns)} user turns, expected 7"
                )


def load_natpress_config(path: Any) -> dict:
    """Read + validate a natpress config JSON file and return it as a dict."""
    import json
    from pathlib import Path

    cfg = json.loads(Path(path).read_text())
    validate_natpress_config(cfg)
    return cfg


# --------------------------------------------------------------------------------------------------
# Arm assembly (single source of truth) + seed derivation
# --------------------------------------------------------------------------------------------------
def assemble_arm_user_turns(scenario: dict, arm: str) -> list[str]:
    """The 7 user turns for one conversation. This is the ONLY place arm layout is defined."""
    t1 = scenario["t1_user"]
    benign = scenario["benign_turns"]
    if arm == "smooth":
        return [t1, *scenario["smooth_turns"], scenario["corner"]]
    if arm == "benign":
        return [t1, *benign]
    if arm == "step":
        return [t1, *benign[0:4], scenario["step_t6"], scenario["corner"]]
    if arm == "latedump":
        return [t1, *benign[0:3], scenario["latedump_t5"], scenario["latedump_t6"], scenario["corner"]]
    raise ValueError(f"unknown arm: {arm!r}")


def shared_turn_indices(arm: str) -> frozenset[int]:
    return _SHARED_TURN_INDICES.get(arm, frozenset())


def prefix_sha(token_ids: Any) -> str:
    """sha256 hex of the comma-joined prompt token ids (the full prompt before an assistant reply)."""
    joined = ",".join(str(int(i)) for i in token_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def turn_seed(protocol: str, scenario_id: str, sample_index: int, turn_index: int,
              prompt_prefix_sha256: str) -> int:
    """Big-endian int of the first 8 bytes of sha256 over the pipe-joined seed key."""
    key = f"{protocol}|{scenario_id}|{sample_index}|{turn_index}|{prompt_prefix_sha256}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def plan_conversations(cfg: dict) -> list[dict]:
    """Deterministic conversation plan; every benign precedes the step/latedump siblings it feeds."""
    spa = cfg["samples_per_arm"]
    offset = int(cfg.get("sample_index_offset", 0))
    plan: list[dict] = []
    for scenario in cfg["scenarios"]:
        sid = scenario["scenario_id"]
        family = scenario["family"]
        for arm in _PLAN_ARM_ORDER:
            for i in range(int(spa[arm])):
                sample_index = offset + i
                plan.append({
                    "conversation_id": f"natpress:{family}:{sid}:{arm}:s{sample_index:02d}",
                    "family": family,
                    "scenario_id": sid,
                    "arm": arm,
                    "sample_index": sample_index,
                    "user_turns": assemble_arm_user_turns(scenario, arm),
                })
    return plan


def _sampling_block(cfg: dict) -> dict:
    sampling = cfg["sampling"]
    return {
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "max_new_tokens_per_turn": int(sampling["max_new_tokens_per_turn"]),
    }


# --------------------------------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------------------------------
class StubGenerator:
    """Deterministic torch-free generator; replies derive from the turn_seed and carry stub=true."""

    is_stub = True

    @staticmethod
    def _serialize(messages: list[dict], *, add_generation_prompt: bool) -> str:
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def prompt_token_ids(self, messages: list[dict]) -> list[int]:
        return list(self._serialize(messages, add_generation_prompt=True).encode("utf-8"))

    def full_token_ids(self, messages: list[dict]) -> list[int]:
        return list(self._serialize(messages, add_generation_prompt=False).encode("utf-8"))

    def decode(self, token_ids: Any) -> str:
        return bytes(int(t) & 0xFF for t in token_ids).decode("utf-8", errors="replace")

    def generate(self, messages: list[dict], seed: int, sampling: dict) -> tuple[str, list[int], str]:
        marker = seed % 997
        text = (
            f"Stub reply {marker} acknowledging the latest turn. "
            f"This deterministic placeholder stands in for a model generation seeded at {seed}. "
            f"No live model produced these words."
        )
        return text, list(text.encode("utf-8")), "stop"


def _sample_top_p(probs: Any, top_p: float, generator: Any) -> Any:
    import torch

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    drop = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_idx.gather(-1, choice)


class HFGenerator:
    """transformers generation glue: per-turn torch.Generator, nucleus sampling, batch of 1."""

    is_stub = False

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        eos_ids = {tokenizer.eos_token_id}
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int):
            eos_ids.add(eot)
        self.eos_ids = {int(e) for e in eos_ids if isinstance(e, int) and e >= 0}

    def prompt_token_ids(self, messages: list[dict]) -> list[int]:
        from geoprobe.models.tokenization import normalize_token_ids

        raw = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return normalize_token_ids(raw)

    def full_token_ids(self, messages: list[dict]) -> list[int]:
        from geoprobe.models.tokenization import normalize_token_ids

        raw = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return normalize_token_ids(raw)

    def decode(self, token_ids: Any) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def generate(self, messages: list[dict], seed: int, sampling: dict) -> tuple[str, list[int], str]:
        import torch

        prompt_ids = self.prompt_token_ids(messages)
        cur = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        rng = torch.Generator(device=cur.device).manual_seed(int(seed) & ((1 << 63) - 1))
        temperature = max(float(sampling["temperature"]), 1e-6)
        top_p = float(sampling["top_p"])
        max_new = int(sampling["max_new_tokens_per_turn"])
        generated: list[int] = []
        finish_reason = "length"
        past = None
        with torch.no_grad():
            for _ in range(max_new):
                out = self.model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :].float() / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = _sample_top_p(probs, top_p, rng)
                token = int(next_id.item())
                generated.append(token)
                if token in self.eos_ids:
                    finish_reason = "stop"
                    break
                cur = next_id.view(1, 1)
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, generated, finish_reason


# --------------------------------------------------------------------------------------------------
# Conversation generation
# --------------------------------------------------------------------------------------------------
def generate_conversation(entry: dict, scenario: dict, cfg: dict, generator: Any,
                          benign_cache: dict, *, model_id: str, config_sha256: str,
                          provenance: dict) -> dict:
    """Roll out one conversation into a ``natpress_rollout_v1`` row, honouring sibling sharing."""
    protocol = cfg["protocol"]
    sampling = _sampling_block(cfg)
    arm = entry["arm"]
    sid = entry["scenario_id"]
    sample_index = entry["sample_index"]
    shared = shared_turn_indices(arm)

    messages: list[dict] = [{"role": "system", "content": scenario["system"]}]
    turns: list[dict] = []
    for turn_index, user_text in enumerate(entry["user_turns"]):
        messages.append({"role": "user", "content": user_text})
        prompt_ids = generator.prompt_token_ids(messages)
        prefix_hex = prefix_sha(prompt_ids)
        seed = turn_seed(protocol, sid, sample_index, turn_index, prefix_hex)
        if turn_index in shared:
            sibling = benign_cache[(sid, sample_index)]
            sibling_turn = sibling["turns"][turn_index]
            assistant_text = sibling_turn["assistant_text"]
            assistant_token_ids = list(sibling_turn["assistant_token_ids"])
            finish_reason = sibling_turn["finish_reason"]
            shared_from = f"benign:s{sample_index:02d}"
        else:
            assistant_text, assistant_token_ids, finish_reason = generator.generate(
                messages, seed, sampling
            )
            shared_from = None
        messages.append({"role": "assistant", "content": assistant_text})
        turns.append({
            "turn_index": turn_index,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "assistant_token_ids": list(assistant_token_ids),
            "prompt_prefix_sha256": prefix_hex,
            "turn_seed": seed,
            "finish_reason": finish_reason,
            "shared_from": shared_from,
        })

    token_ids = list(generator.full_token_ids(messages))
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "conversation_id": entry["conversation_id"],
        "family": entry["family"],
        "scenario_id": sid,
        "arm": arm,
        "sample_index": sample_index,
        "stub": bool(generator.is_stub),
        "model_id": model_id,
        "sampling": sampling,
        "system": scenario["system"],
        "ground_truth": {
            "fact": scenario["ground_truth"]["fact"],
            "honest_verdict": scenario["ground_truth"]["honest_verdict"],
            "deceptive_verdict": scenario["ground_truth"]["deceptive_verdict"],
        },
        "turns": turns,
        "token_ids": token_ids,
        "decoded_text_exact": generator.decode(token_ids),
        "config_sha256": config_sha256,
        "git_provenance": provenance,
    }
    if arm == "benign":
        benign_cache[(sid, sample_index)] = row
    return row
