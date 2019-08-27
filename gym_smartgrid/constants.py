### CASE HEADERS ###
headers_bus = ['BUS_I', 'BUS_TYPE', 'GS', 'BS', 'BASE_KV', 'VMAX', 'VMIN']
BUS_H = dict(zip(headers_bus, range(len(headers_bus))))
headers_dev = ['BUS_I', 'DEV_TYPE', 'Q/P', 'QMAX', 'QMIN',
               'DEV_STATUS', 'PMAX', 'PMIN', 'PC1', 'PC2', 'QC1MIN',
               'QC1MAX', 'QC2MIN', 'QC2MAX', 'SOC_MAX', 'EFF']
DEV_H = dict(zip(headers_dev, range(len(headers_dev))))
headers_branch = ['F_BUS', 'T_BUS', 'BR_R', 'BR_X', 'BR_B', 'RATE_A',
                  'TAP', 'SHIFT', 'BR_STATUS']
BRANCH_H = dict(zip(headers_branch, range(len(headers_branch))))

### Operating ranges of the network. ###
GRID_SPECS = ['PMIN_BUS', 'PMAX_BUS', 'QMIN_BUS', 'QMAX_BUS', 'VMIN_BUS',
              'VMAX_BUS', 'PMIN_DEV', 'PMAX_DEV', 'QMIN_DEV', 'QMAX_DEV',
              'DEV_TYPE', 'RATE', 'SOC_MIN', 'SOC_MAX']

### Rendered values ###
RENDERED_NETWORK_SPECS = ['DEV_TYPE', 'PMIN_DEV', 'PMAX_DEV', 'RATE', 'SOC_MAX']
RENDERED_STATE_VALUES = ['P_DEV', 'P_BR', 'SOC']
