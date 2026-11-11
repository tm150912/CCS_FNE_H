"""Small, self-contained instance for checking the complete Bundle workflow."""
from experiment_setup import NetworkData, compute_cost_data, period_schedule


def build_tiny_instance():
    """Return a two-period network with indivisible capacity and a positive LP gap."""
    nodes = {'H1': [0.0, 0.0], 'I1': [1.0, 0.0], 'P1': [2.0, 0.0]}
    arcs = [('H1', 'I1'), ('I1', 'H1'), ('I1', 'P1'), ('P1', 'I1')]
    diameters = ['small', 'large']
    units = [('H1', 'u1'), ('H1', 'u2')]
    arc_classes = [(u, v, d) for u, v in arcs for d in diameters]
    network = NetworkData(
        ij=arcs, ijd=arc_classes, ig=units,
        sources={'H1': nodes['H1']}, sinks={'P1': nodes['P1']},
        intersections={'I1': nodes['I1']},
        successors={n: [v for u, v in arcs if u == n] for n in nodes},
        predecessors={n: [u for u, v in arcs if v == n] for n in nodes},
        source_units={'H1': ['u1', 'u2']},
        source_capacity={key: 6.0 for key in units},
        pipe_capacity={'small': 6.0, 'large': 12.0},
        max_wells={'P1': 3}, well_capacity={'P1': 5.0},
        source_fixed_cost={key: 100.0 for key in units},
        pipe_fixed_cost={key: 30.0 if key[2] == 'small' else 50.0
                         for key in arc_classes},
        sink_fixed_cost={'P1': 40.0}, well_fixed_cost={'P1': 25.0},
        source_variable_cost={key: 1.0 for key in units},
        pipe_variable_cost={key: 0.25 for key in arc_classes},
        sink_variable_cost={'P1': 1.0})
    periods, durations = period_schedule(2)
    costs = compute_cost_data(network, periods, durations,
                              maximum_capture_amount=8.0)
    return network, costs, diameters, 3
