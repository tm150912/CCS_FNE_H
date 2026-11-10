"""Sparse LR search signals; none of these rankings is a primal saving certificate."""
from collections import defaultdict, deque

import numpy as np

from heuristic_stages.common import COST_FIELDS, Plan, _objective


CONSTRUCTION = ("s", "r", "w", "y")


def construction(plan):
    return Plan(**{name: dict(getattr(plan, name)) for name in CONSTRUCTION})


def _object_values(plan, periods):
    result = {}
    positions = {t: index for index, t in enumerate(periods)}
    for name in CONSTRUCTION:
        for key, value in getattr(plan, name).items():
            obj = (name, key[:-1])
            if obj not in result:
                result[obj] = np.zeros(len(periods))
            result[obj][positions[key[-1]]] += value
    for values in result.values():
        np.cumsum(values, out=values)
    return result


class Guidance:
    def __init__(self, config):
        self.config = config
        self.recent = deque(maxlen=config.guidance_window)
        self.count = 0
        self.last_ub = None
        self.stagnation = 0

    def update(self, ctx, oracle, aggregate=None, *, advance=True, rotation_step=None):
        """Refresh against the incumbent; only new oracle calls enter statistics.

        In the primal-only phase, transport arc regrets are cached ranking
        signals from the last oracle call, not fresh saving certificates.
        """
        ctx.check_time()
        if advance:
            self.count += 1
        if self.last_ub is None or ctx.best < self.last_ub - (
                ctx.options.improvement_tolerance * max(1, abs(self.last_ub))):
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.last_ub = ctx.best
        current = construction(oracle.plan)
        if advance:
            self.recent.append(current)
        suggested = construction(aggregate if aggregate is not None else current)
        incumbent = ctx.incumbent
        base, target = _object_values(incumbent, ctx.t), _object_values(suggested, ctx.t)
        metrics = {name: defaultdict(float) for name in
                   ("cumulative_difference", "selection_frequency", "volatility",
                    "residual", "original_cost", "transport_build_gap", "transport_flow_gap")}
        for obj in base.keys() | target.keys():
            metrics["cumulative_difference"][obj] = float(np.sum(np.abs(
                base.get(obj, 0) - target.get(obj, 0))))
        # Use second moments as well as frequencies: always-one differs from 0/2.
        sums, squares = {}, {}
        for plan in self.recent:
            for obj, values in _object_values(plan, ctx.t).items():
                if obj not in sums:
                    sums[obj], squares[obj] = np.zeros(len(ctx.t)), np.zeros(len(ctx.t))
                sums[obj] += values
                squares[obj] += values * values
                metrics["selection_frequency"][obj] += float(np.count_nonzero(values)) / len(self.recent)
        for obj in sums:
            variance = squares[obj] / len(self.recent) - (sums[obj] / len(self.recent)) ** 2
            metrics["volatility"][obj] = max(0.0, float(np.sum(variance)))
        positions = {node: index for index, node in enumerate(ctx.nodes)}
        residual = np.asarray(oracle.residual)
        objects = set(metrics["cumulative_difference"]) | set(metrics["selection_frequency"])
        for name, prefix in objects:
            endpoints = prefix[:2] if name == "y" else prefix[:1]
            metrics["residual"][name, prefix] = sum(
                float(np.sum(np.abs(residual[positions[node]]))) for node in endpoints)
        for name in (*CONSTRUCTION, "a", "b", "x"):
            construction_name = {"a": "s", "b": "r", "x": "y"}.get(name, name)
            for key, value in getattr(incumbent, name).items():
                metrics["original_cost"][construction_name, key[:-1]] += (
                    getattr(ctx.c, COST_FIELDS[name])[key] * value)
        arc_details = oracle.blocks.get("transport", {}).get("arc_details", {})
        arc_diameters = defaultdict(set)
        for plan in (incumbent, current):
            for key, value in plan.y.items():
                if value:
                    arc_diameters[key[:2]].add(key[2])
        for edge, detail in arc_details.items():
            # Arc regret is shared across diameters. Only existing/suggested
            # objects (or one representative) need ranking entries.
            for d in arc_diameters.get(edge, (ctx.d[0],)):
                obj = ("y", (*edge, d))
                metrics["transport_build_gap"][obj] = detail["build_regret"]
                metrics["transport_flow_gap"][obj] = detail["flow_regret"]
                metrics["residual"][obj] = sum(
                    float(np.sum(np.abs(residual[positions[n]]))) for n in edge)

        # Facility blocks stay whole. Their gaps choose additional effort, never
        # become purported nonnegative per-facility savings.
        block_lagrangian = dict(capture=0.0, storage=0.0, transport=0.0)
        prices = np.asarray(oracle.multipliers)
        period_pos = {t: index for index, t in enumerate(ctx.t)}
        incumbent_residual = np.zeros_like(residual)
        for name in (*CONSTRUCTION, "a", "b", "x"):
            block = "capture" if name in ("s", "a") else "storage" if name in ("r", "w", "b") else "transport"
            for key, value in getattr(incumbent, name).items():
                price = getattr(ctx.c, COST_FIELDS[name])[key]
                h = period_pos[key[-1]]
                if name == "a":
                    price -= prices[positions[key[0]], h]
                    incumbent_residual[positions[key[0]], h] -= value
                elif name == "b":
                    price += prices[positions[key[0]], h]
                    incumbent_residual[positions[key[0]], h] += value
                elif name == "x":
                    price += prices[positions[key[0]], h] - prices[positions[key[1]], h]
                    incumbent_residual[positions[key[0]], h] += value
                    incumbent_residual[positions[key[1]], h] -= value
                block_lagrangian[block] += price * value
        gaps = {block: {"lower": value - oracle.blocks[block]["u"],
                        "upper": None if oracle.blocks[block]["ell"] is None else
                        value - oracle.blocks[block]["ell"]}
                for block, value in block_lagrangian.items()}
        largest = max(gaps, key=lambda block: gaps[block]["lower"])
        reasons = defaultdict(set)
        for metric, scores in metrics.items():
            for kind in CONSTRUCTION:
                count = self.config.guidance_top_k
                block = "capture" if kind == "s" else "transport" if kind == "y" else "storage"
                if block == largest:
                    count += 1
                ranked = sorted((obj for obj, val in scores.items() if obj[0] == kind and val > 0),
                                key=lambda obj: (-scores[obj], str(obj)))
                for obj in ranked[:count]:
                    reasons[obj].add(metric)
                    if block == largest:
                        reasons[obj].add("largest_block_gap")
        # Rotate over the full original domain, including previously unused sites.
        for kind, domain in (("s", ctx.n.ig), ("r", [(j,) for j in ctx.n.sinks]),
                             ("y", [(*e, ctx.d[0]) for e in ctx.n.ij])):
            if domain:
                step = self.count if rotation_step is None else rotation_step
                reasons[kind, tuple(domain[(step - 1) % len(domain)])].add("rotation")
        selected = set(reasons)
        # Opening and wells use coordinated windows.
        for kind, prefix in list(selected):
            if kind in ("r", "w"):
                partner = ("w" if kind == "r" else "r", prefix)
                selected.add(partner)
                reasons[partner].add("site_well_link")
        facilities = {prefix[0] for kind, prefix in selected if kind != "y"}
        arcs = {prefix[:2] for kind, prefix in selected if kind == "y"}
        ctx.check_time()
        return dict(objects=selected, facilities=facilities, arcs=arcs,
                    reasons=dict(reasons), metrics=metrics, suggested=suggested,
                    block_gaps=gaps,
                    gap_identity_error=sum(v["lower"] for v in gaps.values()) -
                    (_objective(ctx.c, incumbent) + float(np.sum(prices * incumbent_residual)) - oracle.u),
                    evaluation=self.count, stagnation=self.stagnation)
