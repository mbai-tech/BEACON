"""VLMWeightUpdater — queries Qwen to propose PlannerConfig updates from SceneSummary diagnostics."""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from typing import Optional

import numpy as np

from beacon.core.models import SceneSummary
from beacon.planning.beacon_planner import PlannerConfig

_SYSTEM_PROMPT = (
    "You are tuning a motion planner with cost "
    "J = W_P·J_pos + w_r(b)·J_risk + w_v(b)·J_vel + W_B·J_resource.\n"
    "Battery-coupled terms — physics constraints you must respect:\n\n"
    "w_r = w_r_scale · b: Risk weight shrinks as battery drains. This reflects the "
    "principle from energy-aware planetary rover planning that when battery is low, "
    "the robot can no longer afford conservative detours — risk tolerance must increase. "
    "Do not increase w_r_scale if the robot frequently ran out of battery mid-scene.\n"
    "w_v = w_v_floor + w_v_range · (1-b): Velocity penalty grows as battery drains. "
    "This is grounded in kinetic energy safety literature: at low battery, the robot "
    "has less capacity to absorb a high-energy collision, so ΔE = 0.5·speed² must be "
    "penalized harder. If mean_speed_at_contact is high and b was low at those events, "
    "increase w_v_range. "
    "If low_battery_contact_fraction is high, the robot is making risky contacts "
    "precisely when its energy budget for absorbing them is smallest — this is the "
    "regime where w_v_range should be highest, per kinetic energy budget literature.\n"
    "delta_E_coeff: Scales the 0.5·speed² contact energy term. This is physically the "
    "dissipated kinetic energy in an inelastic impact — do not reduce it below 0.3, as "
    "this would underestimate collision damage at any battery level.\n\n"
    "Resource decomposition — three-component energy model:\n\n"
    "resource_d, resource_T, resource_contact weight distance, time, and contact energy "
    "respectively, corresponding to the motor/friction, control/sensor overhead, and "
    "collision dissipation components of a robot's energy budget. If contact energy "
    "dominated J_resource, increase resource_contact. If the robot took unnecessarily "
    "long paths, increase resource_d.\n\n"
    "Risk sub-weights:\n\n"
    "geo_weight, sem_weight, dir_weight must sum to 1.0. If semantic damage is high, "
    "increase sem_weight. If the robot frequently had near-collisions (low t_c), "
    "increase geo_weight.\n\n"
    "General constraints: all λ values positive, max 20% change per parameter per "
    "update, simplex constraints on sub-weight groups. Return only valid JSON with "
    "the same keys as input config."
)

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def _config_to_dict(cfg: PlannerConfig) -> dict:
    return {k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in asdict(cfg).items()}


def _summary_to_dict(s: SceneSummary) -> dict:
    d = asdict(s)
    return {
        k: (float(v) if isinstance(v, (float, np.floating)) else
            (int(v)   if isinstance(v, (int,   np.integer))  else v))
        for k, v in d.items()
        if k != "battery_contact_log"
    }


def _summary_to_prompt_dict(s: SceneSummary) -> dict:
    base = _summary_to_dict(s)
    battery_log = list(s.battery_contact_log or [])
    contact_events = [e for e in battery_log if e.get("event") == "contact"]
    stuck_events = [e for e in battery_log if e.get("event") == "stuck"]

    def _mean_event(events: list[dict], key: str) -> float | None:
        vals = [float(e[key]) for e in events if key in e]
        return float(np.mean(vals)) if vals else None

    def _tail_events(events: list[dict], n: int = 3) -> list[dict]:
        trimmed = []
        for event in events[-n:]:
            trimmed.append({
                "event": event.get("event"),
                "b": float(event["b"]) if "b" in event else None,
                "speed": float(event["speed"]) if "speed" in event else None,
                "w_r": float(event["w_r"]) if "w_r" in event else None,
                "w_v": float(event["w_v"]) if "w_v" in event else None,
            })
        return trimmed

    base["battery_contact_log"] = {
        "n_events": len(battery_log),
        "n_contact_events": len(contact_events),
        "n_stuck_events_logged": len(stuck_events),
        "mean_contact_battery": _mean_event(contact_events, "b"),
        "mean_contact_speed": _mean_event(contact_events, "speed"),
        "mean_contact_w_r": _mean_event(contact_events, "w_r"),
        "mean_contact_w_v": _mean_event(contact_events, "w_v"),
        "mean_stuck_battery": _mean_event(stuck_events, "b"),
        "recent_contact_events": _tail_events(contact_events),
        "recent_stuck_events": _tail_events(stuck_events),
        "note": "Compact summary of the full battery/contact trace to keep prompts within model context limits.",
    }
    return base


def _to_json(obj: dict) -> str:
    return json.dumps(obj, indent=2, cls=_Encoder)


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """Return the first JSON object found in text, stripping markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


# ── Post-processing: clip and validate ────────────────────────────────────────

_BATTERY_KEYS        = frozenset({"w_r_scale", "w_v_floor", "w_v_range"})

_DELTA_E_MIN         = 0.3
_MAX_CHANGE          = 0.20       # ±20 % per parameter — scene-level updates
_MAX_CHANGE_FAMILY   = 0.40       # ±40 % per parameter — family-level resets
_RISK_KEYS           = ("geo_weight", "sem_weight", "dir_weight")
_RESOURCE_KEYS       = ("resource_d", "resource_T", "resource_contact")

_FAMILY_HINTS: dict = {
    "sparse": (
        "Battery is rarely a constraint here. w_r_scale can be higher since the robot "
        "can afford risk-averse detours. w_v_range matters less."
    ),
    "sparse_clutter": (
        "Battery is rarely a constraint here. w_r_scale can be higher since the robot "
        "can afford risk-averse detours. w_v_range matters less."
    ),
    "cluttered": (
        "Many contact events expected. delta_E_coeff and resource_contact are the most "
        "important terms. sem_weight should be high."
    ),
    "dense_clutter": (
        "Many contact events expected. delta_E_coeff and resource_contact are the most "
        "important terms. sem_weight should be high."
    ),
    "semantic_trap": (
        "The robot will contact mislabeled fragile objects. After CIBP corrects beliefs, "
        "sem_weight drives replanning. Prioritize it over geo_weight."
    ),
    "collision_shortcut": (
        "The robot will contact mislabeled fragile objects. After CIBP corrects beliefs, "
        "sem_weight drives replanning. Prioritize it over geo_weight."
    ),
    "perturbed": (
        "Battery state at perturbation time varies. w_r_scale should be conservative "
        "since the robot cannot predict when a perturbation will force a costly replan."
    ),
    "perturbed_scenes": (
        "Battery state at perturbation time varies. w_r_scale should be conservative "
        "since the robot cannot predict when a perturbation will force a costly replan."
    ),
}


def _aggregate_summaries(summaries: list) -> dict:
    """Aggregate a list of SceneSummary objects into a flat stats dict for the VLM."""
    if not summaries:
        return {}
    n = len(summaries)

    def _mean(vals: list) -> float:
        return float(np.mean(vals)) if vals else 0.0

    all_dominant = [s.dominant_j for s in summaries]
    dominant_j   = max(set(all_dominant), key=all_dominant.count)

    bfs = [s.battery_at_first_stuck for s in summaries
           if s.battery_at_first_stuck is not None]

    return {
        "n_scenes":                        n,
        "success_rate":                    sum(s.success for s in summaries) / n,
        "mean_final_battery":              _mean([s.final_battery             for s in summaries]),
        "mean_total_semantic_damage":      _mean([s.total_semantic_damage     for s in summaries]),
        "mean_forbidden_contact_rate":     _mean([s.forbidden_contact_rate    for s in summaries]),
        "mean_fragile_contact_rate":       _mean([s.fragile_contact_rate      for s in summaries]),
        "mean_j_risk":                     _mean([s.mean_j_risk               for s in summaries]),
        "mean_j_vel":                      _mean([s.mean_j_vel                for s in summaries]),
        "mean_j_resource":                 _mean([s.mean_j_resource           for s in summaries]),
        "mean_n_cibp_replans":             _mean([s.n_cibp_replans            for s in summaries]),
        "mean_n_stuck_events":             _mean([s.n_stuck_events            for s in summaries]),
        "mean_speed_at_contact":           _mean([s.mean_speed_at_contact     for s in summaries]),
        "dominant_j":                      dominant_j,
        "mean_low_battery_contact_fraction": _mean([s.low_battery_contact_fraction for s in summaries]),
        "mean_battery_at_first_stuck":     _mean(bfs) if bfs else None,
    }


def _clip_and_validate(proposed: dict, current: PlannerConfig,
                       max_change: float = _MAX_CHANGE) -> PlannerConfig:
    """
    Enforce all constraints on a VLM-proposed config dict:
      1. Max ±20 % change from current value per parameter.
      2. All values strictly positive.
      3. delta_E_coeff ≥ 0.3.
      4. f_alpha_threshold ∈ (0, π/2).
      5. Renormalise risk and resource sub-weight groups to sum to 1.
    Missing or non-numeric keys fall back to the current value.
    """
    cur  = _config_to_dict(current)
    out  = {}

    for key, cur_val in cur.items():
        if key not in proposed:
            out[key] = cur_val
            continue
        try:
            val = float(proposed[key])
        except (TypeError, ValueError):
            out[key] = cur_val
            continue

        # ① change clamp (scene: ±20 %, family: ±40 %)
        val = max(cur_val * (1.0 - max_change),
                  min(cur_val * (1.0 + max_change), val))
        # ② Strictly positive
        val = max(1e-6, val)
        out[key] = val

    # ③ Hard lower bound on contact-energy coefficient
    out["delta_E_coeff"] = max(_DELTA_E_MIN, out["delta_E_coeff"])

    # ④ f_alpha_threshold ∈ (0, π/2)
    _eps = 1e-6
    out["f_alpha_threshold"] = max(_eps,
                                   min(math.pi / 2.0 - _eps,
                                       out["f_alpha_threshold"]))

    # ⑤ Renormalise simplex groups
    for group in (_RISK_KEYS, _RESOURCE_KEYS):
        total = sum(out[k] for k in group)
        if total > 0:
            for k in group:
                out[k] = out[k] / total

    return PlannerConfig(**out)


# ── VLMWeightUpdater ──────────────────────────────────────────────────────────

class VLMWeightUpdater:
    """
    Calls a Qwen model (via OpenAI-compatible API) to propose updated
    PlannerConfig weights after each scene, then clips and validates the result.

    Environment variables (used as fallbacks):
      QWEN_BASE_URL     — API base URL   (default: DashScope compatible endpoint)
      DASHSCOPE_API_KEY — API key
    """

    def __init__(
        self,
        model:       str           = "qwen-plus",
        base_url:    Optional[str] = None,
        api_key:     Optional[str] = None,
        temperature: float         = 0.2,
        max_tokens:  int           = 512,
    ):
        self.model       = model
        self.base_url    = base_url or os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key     = api_key  or os.getenv("DASHSCOPE_API_KEY") or "local"
        self.temperature = temperature
        self.max_tokens  = max_tokens

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_user_message(
        self,
        config:              PlannerConfig,
        summary:             SceneSummary,
        history:             list[tuple[PlannerConfig, SceneSummary]],
        family:              str = "unknown",
        scene_idx_in_family: int = 0,
        scenes_remaining:    int = 0,
        history_limit:       int = 5,
    ) -> str:
        parts: list[str] = []

        tail = history[-history_limit:] if history_limit > 0 else []
        if tail:
            parts.append("=== Recent history (oldest → newest) ===")
            for i, (hcfg, hsum) in enumerate(tail, 1):
                parts.append(f"--- Entry {i} ---")
                parts.append(f"config:\n{_to_json(_config_to_dict(hcfg))}")
                parts.append(f"summary:\n{_to_json(_summary_to_prompt_dict(hsum))}")

        parts.append("=== Current config ===")
        parts.append(_to_json(_config_to_dict(config)))

        parts.append("=== Latest scene summary ===")
        parts.append(_to_json(_summary_to_prompt_dict(summary)))

        parts.append("=== Optimization context ===")
        parts.append(_to_json({
            "current_family":            family,
            "scene_index_within_family": scene_idx_in_family,
            "scenes_remaining_in_family": scenes_remaining,
        }))
        parts.append(
            f"You are optimizing exclusively for {family} scenes. Do not generalize. "
            "If scenes_remaining_in_family > 10, allow up to 25% change per parameter. "
            "If \u2264 5, limit to 10% \u2014 you are converging, not exploring."
        )

        parts.append(
            "Based on the diagnostics above and the physics constraints in the "
            "system prompt, return an updated config JSON with exactly the same "
            "keys. Return ONLY the JSON object — no explanation, no markdown fences."
        )
        return "\n\n".join(parts)

    # ── Family-level prompt construction ──────────────────────────────────────

    def _build_family_message(
        self,
        config:     PlannerConfig,
        aggregated: dict,
        family:     str,
        history:    list[tuple[PlannerConfig, "SceneSummary"]],
        history_limit: int = 5,
    ) -> str:
        parts: list[str] = []

        hint = _FAMILY_HINTS.get(family, "")
        if hint:
            parts.append(f"=== Family hint: {family} ===")
            parts.append(hint)

        tail = history[-history_limit:] if history_limit > 0 else []
        if tail:
            parts.append("=== Recent per-scene history (oldest → newest) ===")
            for i, (hcfg, hsum) in enumerate(tail, 1):
                parts.append(f"--- Entry {i} ---")
                parts.append(f"config:\n{_to_json(_config_to_dict(hcfg))}")
                parts.append(f"summary:\n{_to_json(_summary_to_prompt_dict(hsum))}")

        parts.append("=== Current config ===")
        parts.append(_to_json(_config_to_dict(config)))

        parts.append("=== Family-aggregated stats ===")
        parts.append(_to_json(aggregated))

        parts.append(
            "Based on the family-level diagnostics above and the physics constraints "
            "in the system prompt, return an updated config JSON with exactly the same "
            "keys. This is a family-level reset — you may change parameters by up to "
            "40%. Return ONLY the JSON object — no explanation, no markdown fences."
        )
        return "\n\n".join(parts)

    def _call_llm_with_retries(self, client, build_messages) -> str:
        last_exc: Exception | None = None

        for history_limit in (5, 3, 1, 0):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=build_messages(history_limit),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if any(token in msg for token in (
                    "maximum context length",
                    "context length",
                    "prompt contains at least",
                    "input tokens",
                    "too long",
                )):
                    continue
                break

        if last_exc is not None:
            print(f"  Warning: VLM update skipped after prompt-size retries: {last_exc}")
        raise last_exc if last_exc is not None else RuntimeError("Unknown VLM call failure")

    # ── Main API call ──────────────────────────────────────────────────────────

    def update(
        self,
        config:              PlannerConfig,
        summary:             SceneSummary,
        history:             list[tuple[PlannerConfig, SceneSummary]],
        family:              str  = "unknown",
        scene_idx_in_family: int  = 0,
        scenes_remaining:    int  = 0,
        battery_only:        bool = False,
    ) -> PlannerConfig:
        """
        Query Qwen with the current config, latest SceneSummary, and the last
        five (config, summary) history pairs.  Parse the response, clip every
        value to an adaptive bound (25% when scenes_remaining > 10, 10% when
        ≤ 5, 20% otherwise), enforce physical hard bounds, and renormalise
        simplex groups.  Returns current config unchanged if the model response
        cannot be parsed.

        If battery_only=True (condition C), only w_r_scale / w_v_floor /
        w_v_range are applied from the VLM proposal; all other parameters revert
        to their values in ``config``.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        if scenes_remaining > 10:
            max_change = 0.25
        elif scenes_remaining <= 5:
            max_change = 0.10
        else:
            max_change = _MAX_CHANGE

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            raw = self._call_llm_with_retries(
                client,
                lambda history_limit: [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_user_message(
                        config, summary, history,
                        family=family,
                        scene_idx_in_family=scene_idx_in_family,
                        scenes_remaining=scenes_remaining,
                        history_limit=history_limit,
                    )},
                ],
            )
        except Exception:
            return config

        proposed = _extract_json(raw)

        if proposed is None:
            return config

        new_config = _clip_and_validate(proposed, config, max_change=max_change)

        if battery_only:
            cur = _config_to_dict(config)
            new = _config_to_dict(new_config)
            for k in cur:
                if k not in _BATTERY_KEYS:
                    new[k] = cur[k]
            new_config = PlannerConfig(**new)

        return new_config

    def update_family(
        self,
        config:    PlannerConfig,
        summaries: list,
        family:    str,
        history:   list[tuple[PlannerConfig, "SceneSummary"]],
    ) -> PlannerConfig:
        """
        Query Qwen with family-aggregated stats from all scenes in a family.
        Uses ±40 % change bounds instead of ±20 %.  Returns current config
        unchanged if summaries is empty or the model response cannot be parsed.
        """
        aggregated = _aggregate_summaries(summaries)
        if not aggregated:
            return config

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        try:
            raw = self._call_llm_with_retries(
                client,
                lambda history_limit: [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": self._build_family_message(
                        config, aggregated, family, history, history_limit=history_limit,
                    )},
                ],
            )
        except Exception:
            return config

        proposed = _extract_json(raw)

        if proposed is None:
            return config

        return _clip_and_validate(proposed, config, max_change=_MAX_CHANGE_FAMILY)
