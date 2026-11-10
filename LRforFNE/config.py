"""Algorithm defaults; original model/data settings remain in experiment_setup."""
from dataclasses import dataclass, asdict
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class LRConfig:
    lr_time_limit: float = 300.0
    max_iterations: int = 100  # oracle cap including refinements; primal search shares lr_time_limit
    target_gap: float = 1e-4
    oracle_time_limit: float = 30.0
    capture_mip_gap: float = 2e-4
    storage_mip_gap: float = 2e-4
    oracle_abs_tol: float = 1e-6
    oracle_rel_tol: float = 1e-9
    oracle_refinements: int = 2
    bundle_rho: Optional[float] = None
    bundle_rho_floor: float = 1e-8
    bundle_eta: float = 0.1
    bundle_tolerance: float = 1e-8
    bundle_adaptive_rho: bool = True
    bundle_rho_growth: float = 2.0
    bundle_rho_shrink: float = 0.7
    bundle_rho_max_factor: float = 1e4
    bundle_uncertain_limit: int = 5
    bundle_diagnostic_every: int = 20
    bundle_diagnostic_time_limit: float = 5.0
    delta_cost: float = 1.0
    feasibility_tolerance: float = 1e-6
    eps_improve: float = 1e-7
    threads: int = 1
    seed: int = 0
    gurobi_output_flag: bool = False
    output_flag: bool = True
    neighborhoods_enabled: bool = True
    neighborhood_every: int = 5
    neighborhood_time_limit: float = 30.0
    neighborhood_mip_gap: float = 1e-4
    flow_time_limit: float = 10.0
    guidance_window: int = 5
    guidance_top_k: int = 3
    n1_max_moves: int = 8
    n2_tau_minus: int = 1
    n2_tau_plus: int = 1
    n3_max_arcs: int = 80
    n3_max_facilities: int = 12
    n3_candidate_limit: int = 6
    n3_expand_corridors: int = 4  # 0 retains the original N3 domains
    n3_expanded_time_limit: float = 45.0
    pipeline_polish_time_limit: float = 40.0  # 0 disables LR pipe polish
    n3_refine_candidates: int = 1
    n3_refine_time_limit: float = 1.0
    n3_refine_absolute_gap: float = 100.0

    def validate(self):
        for name in ('lr_time_limit', 'oracle_time_limit', 'delta_cost',
                     'bundle_rho_floor', 'feasibility_tolerance',
                     'neighborhood_time_limit', 'flow_time_limit',
                     'n3_expanded_time_limit',
                     'n3_refine_time_limit', 'n3_refine_absolute_gap',
                     'bundle_diagnostic_time_limit'):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('target_gap', 'oracle_abs_tol', 'oracle_rel_tol',
                     'bundle_tolerance', 'eps_improve', 'neighborhood_mip_gap',
                     'capture_mip_gap', 'storage_mip_gap',
                     'pipeline_polish_time_limit'):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        for name in ('max_iterations', 'neighborhood_every',
                     'guidance_window', 'guidance_top_k', 'n3_max_arcs', 'n3_max_facilities',
                     'n3_candidate_limit', 'bundle_uncertain_limit'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        for name in ('oracle_refinements', 'threads', 'seed', 'n1_max_moves',
                     'n2_tau_minus', 'n2_tau_plus', 'n3_refine_candidates',
                     'bundle_diagnostic_every', 'n3_expand_corridors'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if not 0 < self.bundle_eta < 1:
            raise ValueError('Require 0<bundle_eta<1')
        if self.bundle_rho is not None and (not isfinite(self.bundle_rho) or self.bundle_rho <= 0):
            raise ValueError('bundle_rho must be finite and strictly positive')
        if type(self.bundle_adaptive_rho) is not bool:
            raise ValueError('bundle_adaptive_rho must be boolean')
        if not isfinite(self.bundle_rho_growth) or self.bundle_rho_growth <= 1:
            raise ValueError('bundle_rho_growth must be finite and greater than one')
        if not isfinite(self.bundle_rho_shrink) or not 0 < self.bundle_rho_shrink < 1:
            raise ValueError('bundle_rho_shrink must lie strictly between zero and one')
        if not isfinite(self.bundle_rho_max_factor) or self.bundle_rho_max_factor < 1:
            raise ValueError('bundle_rho_max_factor must be finite and at least one')
        return self

    def to_dict(self):
        return asdict(self)

