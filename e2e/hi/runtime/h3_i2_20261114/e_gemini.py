"""Capture Gemini's native-search candidates for the E-stage evaluator."""

import copy
from contextlib import contextmanager
import math
import pickle
import random
import time
from typing import NamedTuple

import numpy as np


class SearchComplete(Exception):
    """The template reached the end of SA and its snapshots are ready."""


class SearchState(NamedTuple):
    blob: bytes
    link_loads: dict
    link_counts: dict
    free_nodes: tuple


def save_state(optimizer):
    platform = optimizer.hardware_platform
    return SearchState(
        pickle.dumps(optimizer.state, protocol=pickle.HIGHEST_PROTOCOL),
        dict(platform.link_loads_dict),
        dict(platform.link_loads_count),
        tuple(optimizer.free_node),
    )


def restore_link_counters(platform, saved):
    platform.link_loads_dict.clear()
    platform.link_loads_dict.update(saved.link_loads)
    platform.link_loads_count.clear()
    platform.link_loads_count.update(saved.link_counts)


def restore_state(optimizer, saved, *, verify=False):
    state = pickle.loads(saved.blob)
    # Keep the template's references to the dictionaries and recover their order.
    optimizer.state[:2] = state[:2]
    for live, restored in zip(optimizer.state[2:], state[2:]):
        live.clear()
        live.update(restored)
    restore_link_counters(optimizer.hardware_platform, saved)
    optimizer.free_node = list(saved.free_nodes)
    if verify:
        # Compare against the decoded state, where set iteration may have changed.
        expected = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return save_state(optimizer) == saved._replace(blob=expected)


def evaluation_snapshot(saved, platform):
    state = pickle.loads(saved.blob)
    platform = copy.deepcopy(platform)
    restore_link_counters(platform, saved)
    return state[5], state[6], state[7], platform


def move_type(optimizer):
    if not optimizer.change_flag:
        return "swap"
    if optimizer.change_mem and optimizer.mem_enable:
        return "ddr"
    return "relocate"


def check_native_revert(optimizer, saved, energy):
    """Exercise the artifact's own undo without altering the search's RNG."""
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        optimizer.revert()
        actual = save_state(optimizer)
        return {
            "energy_equal": bool(optimizer.energy() == energy),
            "state_bytes_equal": actual.blob == saved.blob,
            "link_counters_equal": (actual.link_loads == saved.link_loads
                                    and actual.link_counts == saved.link_counts),
            "free_nodes_equal": actual.free_nodes == saved.free_nodes,
        }
    except Exception as exc:
        # This is an audit of the legacy undo, not the undo used for the search.
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)


@contextmanager
def install(optimizer_class, *, seed, steps=1000, validate_rollback=False,
            capture_initial=False, find_context=None):
    """Intercept the native template's anneal call and capture pre-broadcast states."""
    if find_context is None:
        from e1_fidelity import _find_context
        find_context = _find_context

    original = optimizer_class.anneal
    capture = {}

    def anneal(optimizer):
        if not (optimizer.LP_flag and optimizer.hops_only
                and not optimizer.dijkstra_routing and not optimizer.mem_datatags):
            raise ValueError("expected the GPT Gemini template with LP, hops-only energy and XY routing")
        if not 1 <= steps <= optimizer.steps:
            raise ValueError(f"expected between 1 and {int(optimizer.steps)} Gemini steps")

        context = find_context()
        random.seed(seed)
        np.random.seed(seed)
        started = time.perf_counter()
        temperature_factor = -math.log(optimizer.Tmax / optimizer.Tmin)
        energy = optimizer.energy()
        best_energy, best_step = energy, 0
        before = best = save_state(optimizer)
        if capture_initial:
            capture["initial"] = evaluation_snapshot(before, optimizer.hardware_platform)
        rejected = {"relocate": 0, "swap": 0, "ddr": 0}
        audit = {}

        for step in range(1, steps + 1):
            if optimizer.user_exit:
                raise RuntimeError("Gemini search interrupted before completing its planned steps")
            if step > 1:
                before = save_state(optimizer)
            temperature = optimizer.Tmax * math.exp(temperature_factor * step / steps)
            difference = optimizer.move()
            if difference is None:
                new_energy = optimizer.energy()
                difference = new_energy - energy
            else:
                new_energy = energy + difference

            if difference > 0.0 and math.exp(-difference / temperature) < random.random():
                kind = move_type(optimizer)
                rejected[kind] += 1
                audit_move = validate_rollback and kind not in audit
                if audit_move:
                    audit[kind] = check_native_revert(optimizer, before, energy)
                restored = restore_state(optimizer, before, verify=audit_move)
                if optimizer.energy() != energy or restored is False:
                    raise RuntimeError(f"Gemini rollback did not restore {kind} state at step {step}")
                if audit_move:
                    audit[kind]["adapter_restore_equal"] = restored
            else:
                energy = new_energy
                if energy < best_energy:
                    best_energy, best_step = energy, step
                    best = save_state(optimizer)

        elapsed = time.perf_counter() - started
        platform = optimizer.hardware_platform
        capture["ctx"] = context
        capture["best"] = evaluation_snapshot(best, platform)
        capture["terminal"] = evaluation_snapshot(save_state(optimizer), platform) if seed == 123 else None
        capture["meta"] = {
            "initializer_seed": 123,
            "anneal_seed": seed,
            "steps": steps,
            "best_step": best_step,
            "best_energy": int(best_energy),
            "terminal_energy": int(energy),
            "Tmax": optimizer.Tmax,
            "Tmin": optimizer.Tmin,
            "LP_flag": optimizer.LP_flag,
            "hops_only": optimizer.hops_only,
            "dijkstra_routing": optimizer.dijkstra_routing,
            "rollback": "pre_move_snapshot",
            "sa_wall_s": elapsed,
            "rejections": rejected,
            "native_revert_audit": audit,
        }
        # The E driver evaluates these snapshots with common backend seeds.
        raise SearchComplete

    optimizer_class.anneal = anneal
    try:
        yield capture
    finally:
        optimizer_class.anneal = original
