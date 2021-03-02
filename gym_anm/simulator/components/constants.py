headers_bus = ['BUS_ID', 'BUS_TYPE', 'BASE_KV', 'VMAX', 'VMIN']
headers_dev = ['DEV_ID', 'BUS_ID', 'DEV_TYPE', 'Q/P', 'PMAX', 'PMIN', 'QMAX', 'QMIN',
               'P+', 'P-', 'Q+', 'Q-', 'SOC_MAX', 'SOC_MIN', 'EFF']
headers_branch = ['F_BUS', 'T_BUS', 'BR_R', 'BR_X', 'BR_B', 'RATE', 'TAP', 'SHIFT']

BUS_H = dict(zip(headers_bus, range(len(headers_bus))))
"""Bus column index map in the input dictionary."""

DEV_H = dict(zip(headers_dev, range(len(headers_dev))))
"""Device column index map in the input dictionary."""

BRANCH_H = dict(zip(headers_branch, range(len(headers_branch))))
"""Branch column index map in the input dictionary."""

# Note: the 1st unit is the one used when none is provided (i.e., the default).
STATE_VARIABLES = {'bus_p': ('MW', 'pu'),
                   'bus_q': ('MVAr', 'pu'),
                   'bus_v_magn': ('pu', 'kV'),
                   'bus_v_ang': ('degree', 'rad'),
                   'bus_i_magn': ('pu', 'kA'),
                   'bus_i_ang': ('degree', 'rad'),
                   'dev_p': ('MW', 'pu'),
                   'dev_q': ('MVAr', 'pu'),
                   'des_soc': ('MWh', 'pu'),
                   'gen_p_max': ('MW', 'pu'),
                   'branch_p': ('MW', 'pu'),
                   'branch_q': ('MVAr', 'pu'),
                   'branch_s': ('MVA', 'pu'),
                   'branch_i_magn': ('pu'),
                   'branch_i_ang': ('degree', 'rad'),
                   'aux': (None,)
                  }
"""Variables that can be used in state/observation vectors and their units."""