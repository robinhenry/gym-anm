import cvxpy as cp
import numpy as np


def solve_OPF(Y_bus, full_state):

    # Extract indices.
    bus_ids = list(full_state['bus_p']['pu'].keys())
    dev_ids = list(full_state['dev_p']['pu'].keys())
    branch_ids = list(full_state['branch_p']['pu'].keys())

    # Size of each set.
    N_bus = len(bus_ids)
    N_dev = len(dev_ids)
    N_branch = len(branch_ids)

    # Optimization variables.
    P_bus = cp.Variable(N_bus)
    V_bus_ang = cp.Variable(N_bus)
    P_dev = cp.Variable(N_dev)
    P_branch = cp.Variable(N_branch)

    # Assumption : lines are lossless.
    B_bus = np.imag(Y_bus)

    # Power flow constraints.
    A =
    for (i, j) in branch_ids:








