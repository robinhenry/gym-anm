import numpy as np
import pypsa
from warnings import warn
import logging


logging.getLogger('pypsa').setLevel(logging.WARNING)


def solve_pfe(simulator):
    buses = simulator.buses
    branches = simulator.branches
    devices = simulator.devices

    network = pypsa.Network()

    # Add buses.
    for bus in buses.values():
        network.add('Bus', 'bus {}'.format(bus.id),
                    v_nom=1.)

    # Add branches.
    for branch in branches.values():
        network.add('Transformer',
                    'branch ({}, {})'.format(branch.f_bus, branch.t_bus),
                    bus0='bus {}'.format(branch.f_bus),
                    bus1='bus {}'.format(branch.t_bus),
                    r=branch.r, x=branch.x, g=0, b=branch.b,
                    model='pi', tap_side=0, s_nom=1.,
                    tap_ratio=branch.tap_magn,
                    phase_shift=branch.shift * 180 / np.pi)

    # Add devices.
    for device in devices.values():
        if not device.is_slack:
            network.add('Load',
                        'dev {}'.format(device.dev_id),
                        bus='bus {}'.format(device.bus_id),
                        p_set=-device.p,
                        q_set=-device.q)
        else:
            network.add('Generator',
                        'slack',
                        bus='bus {}'.format(device.bus_id),
                        control='Slack')

    # Solve PFEs.
    network.pf(x_tol=1e-6)

    # Construct V nodal vector.
    V = []
    for bus in buses.values():
        v_magn = network.buses_t.v_mag_pu['bus {}'.format(bus.id)][0]
        v_ang = network.buses_t.v_ang['bus {}'.format(bus.id)][0]
        v = v_magn * np.exp(1.j * v_ang)
        V.append(v)

        # Compute nodal current injections as I = YV.
    I = np.dot(simulator.Y_bus, V)

    # Update simulator.
    for i, bus in enumerate(buses.values()):
        bus.v = V[i]
        bus.i = I[i]

        # Update slack bus/device power injection.
        if bus.is_slack:
            bus.p = network.buses_t.p['bus {}'.format(bus.id)][0]
            bus.q = network.buses_t.q['bus {}'.format(bus.id)][0]

    # Update slack device injections.
    for dev in devices.values():
        if dev.is_slack:
            dev.p = buses[dev.bus_id].p
            dev.q = buses[dev.bus_id].q

            # Warn the user if the slack generation constraints are violated.
            if dev.p > dev.p_max or dev.p < dev.p_min:
                warn('The solution to the PFEs has the slack generator '
                     'inject P=%.2f p.u., outside of its operating range '
                     '[%.d, %d].' % (dev.p, dev.p_min, dev.p_max))
            if dev.q > dev.q_max or dev.q < dev.q_min:
                warn('The solution to the PFEs has the slack generator '
                     'inject Q=%.2f p.u., outside of its operating range '
                     '[%.d, %d].' % (dev.q, dev.q_min, dev.q_max))

    # Compute branch I_{ij}, P_{ij}, and Q_{ij} flows.
    for branch in branches.values():
        v_f = buses[branch.f_bus].v
        v_t = buses[branch.t_bus].v
        branch.compute_currents(v_f, v_t)
        branch.compute_power_flows(v_f, v_t)

    return
