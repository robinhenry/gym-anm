### Network input dictionary headers. ###
headers_bus = ['BUS_ID', 'BUS_TYPE', 'BASE_KV', 'VMAX', 'VMIN']
BUS_H = dict(zip(headers_bus, range(len(headers_bus))))

headers_dev = ['DEV_ID', 'BUS_ID', 'DEV_TYPE', 'Q/P', 'PMAX', 'PMIN', 'QMAX', 'QMIN',
               'P+', 'P-', 'Q+', 'Q-', 'SOC_MAX', 'SOC_MIN', 'EFF']
DEV_H = dict(zip(headers_dev, range(len(headers_dev))))

headers_branch = ['F_BUS', 'T_BUS', 'BR_R', 'BR_X', 'BR_B', 'RATE', 'TAP', 'SHIFT']
BRANCH_H = dict(zip(headers_branch, range(len(headers_branch))))

# ### Operating ranges of the network. ###
# GRID_SPECS = ['bus_p_min', 'bus_p_max', 'bus_q_min', 'bus_q_max',
#               'bus_v_magn_min', 'bus_v_magn_max', 'dev_p_min', 'dev_p_max',
#               'dev_q_min', 'dev_q_max', 'soc_min', 'soc_max', 'branch_s_max']

### Possible variables to use in observation space. ###
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
                   'branch_i_ang': ('rad', 'degree'),
                   'aux': (None,)
                  }
