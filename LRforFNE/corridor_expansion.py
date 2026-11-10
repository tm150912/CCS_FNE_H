"""Enlarge scheduled N3 domains with complete related incumbent flow corridors."""
from heuristic_stages.p5_improvement import RerouteNeighborhood


def _signature(neighborhood):
    return frozenset(neighborhood.arcs), frozenset(neighborhood.facilities)


def expand_scheduled(ctx, scheduled, corridors, config):
    """Keep economic order, original labels and refinement slots after deduplication.

    The returned map associates each original domain with its actual expanded
    domain. Solver-completion and refinement caches use only the latter.
    """
    expanded, indices, domains = [], {}, {}
    sinks = set(ctx.n.sinks)
    for original, refine in scheduled:
        ctx.check_time()
        original_nodes = original.facilities | {node for edge in original.arcs for node in edge}
        original_sinks = original.facilities & sinks
        related = []
        for route in corridors:
            edges = set(zip(route, route[1:]))
            common_nodes = len(set(route) & original_nodes)
            common_arcs = len(edges & original.arcs)
            common_sink = route[-1] in original_sinks
            if common_sink or common_arcs or common_nodes:
                related.append((route, edges, common_sink, common_arcs, common_nodes))
        related.sort(key=lambda item: (-int(item[2]), -item[3], -item[4],
                                      -corridors[item[0]]['score'], str(item[0])))
        arcs, facilities = set(original.arcs), set(original.facilities)
        added, rejected = [], []
        for route, edges, _, _, _ in related:
            ctx.check_time()
            if len(added) >= config.n3_expand_corridors:
                break
            endpoints = {route[0], route[-1]}
            if edges <= arcs and endpoints <= facilities:
                continue
            if (len(arcs | edges) > config.n3_max_arcs
                    or len(facilities | endpoints) > config.n3_max_facilities):
                rejected.append(list(route))
                continue
            arcs.update(edges)
            facilities.update(endpoints)
            added.append(list(route))
        neighborhood = RerouteNeighborhood(arcs, facilities, original.score, original.label)
        signature = _signature(neighborhood)
        domains[_signature(original)] = signature
        details = dict(label=original.label, original_arcs=len(original.arcs),
            original_facilities=len(original.facilities), expanded_arcs=len(arcs),
            expanded_facilities=len(facilities), added_corridors=added,
            rejected_complete_corridors=rejected,
            original_arc_list=[list(edge) for edge in sorted(original.arcs, key=str)],
            expanded_arc_list=[list(edge) for edge in sorted(arcs, key=str)],
            original_facility_list=sorted(original.facilities, key=str),
            expanded_facility_list=sorted(facilities, key=str),
            model_time_limit=config.n3_expanded_time_limit)
        if signature in indices:
            index = indices[signature]
            kept, previous_refine = expanded[index]
            expanded[index] = kept, previous_refine or refine
            ctx.record('LR_N3_expansion', skipped='duplicate_expanded_domain',
                       kept_label=kept.label, refinement_transferred=refine, **details)
            continue
        indices[signature] = len(expanded)
        expanded.append((neighborhood, refine))
        ctx.record('LR_N3_expansion', **details)
    return expanded, domains
