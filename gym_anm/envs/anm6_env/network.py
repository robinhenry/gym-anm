import numpy as np

##### CASE FILE DESCRIPTION #####

### Metadata ###
# baseMVA: base power of the system (MVA).

### 1. Bus data:
# BUS_I: bus number (0-indexing).
# BUS_TYPE: bus type (1 = PQ, 2 = PV, 3 = slack).
# BASE_KV : base voltage of the zone containing the bus (kV).
# VMAX: maximum voltage magnitude (p.u.).
# VMIN: minimum voltage magnitude (p.u.).

### 2. Device data:
# BUS_I: bus number.
# DEV_TYPE: -1 (load), 0 (slack), 1 (power plant), 2 (wind), 3 (solar), 4 (storage).
# Q/P: constant ratio of reactive power over real power.
# QMAX: maximum reactive power output (MVAr).
# QMIN: minimum reactive power output (MVAr).
# DEV_STATUS: machine status (> 0 = in service).
# PMAX: maximum real power output (MW).
# PMIN: minimum real power output (MW).
# PC1: lower real power output of PQ capability curve (MW).
# PC2: upper real power output of PQ capability curve (MW).
# QC1MIN: minimum reactive power output at PC1 (MVAr).
# QC1MAX: maximum reactive power output at PC1 (MVAr).
# QC2MIN: minimum reactive power output at PC2 (MVAr).
# QC2MAX: maximum reactive power output at PC2 (MVAr).
# SOC_MAX: maximum state of charge of storage unit (MWh).
# EFF: round-trip efficiency coefficient of storage unit.

### 3. Branch data.
# F_BUS: "from" bus number.
# T_BUS: "to" bus number.
# BR_R: resistance (p.u.).
# BR_X: reactance (p.u.).
# BR_B: total line charging susceptance (p.u.).
# RATE: MVA rating.
# TAP: transformer off-nominal turns ratio. If non-zero, taps located at
# 'from' bus and impedance at 'to' bus (see pi-model); if zero, indicating
# no-transformer (i_from.e. a simple transmission line).
# SHIFT: transformer phase shit angle (degrees), positive => delay.
# BR_STATUS: branch status, 1 = in service, 0 = out-of-service.

network = {"version": "ANM"}

network["baseMVA"] = 100.0

network["bus"] = np.array([
    [0, 3, 132, 1.04, 1.04],
    [1, 1, 33, 1.1, 0.9],
    [2, 1, 33, 1.1, 0.9],
    [3, 1, 33, 1.1, 0.9],
    [4, 1, 33, 1.1, 0.9],
    [5, 1, 33, 1.1, 0.9]
])

network["device"] = np.array([
    [0,  0,   0., 100, -100,  1, 100, -100,   0,  0,     0,    0,     0,    0,   0,   0],
    [3, -1,  0.2,  0,    0,  1,   0,  -10,   0,  0,     0,    0,     0,    0,   0,   0],
    [3,  3,  0.2, 30,  -30,  1,  30,    0,  20, 30,   -30,   30,   -15,   15,   0,   0],
    [4, -1,  0.2,  0,    0,  1,   0,  -30,   0,  0,     0,    0,     0,    0,   0,   0],
    [4,  2,  0.2,  50,  -50,  1,  50,    0,  35, 50,   -50,   50,   -20,   20,   0,   0],
    [5, -1,  0.2,  0,    0,  1,   0,  -30,   0,  0,     0,    0,     0,    0,   0,   0],
    [5,  4,    0,  50,  -50,  1,  50,  -50,  30, 50,   -50,   50,   -25,   25, 100, 0.9]
])

network["branch"] = np.array([
    [0, 1, 0.0036, 0.1834, 0., 32, 0, 0, 1],         # HV/MV transformer
    [1, 2,   0.03,  0.022, 0., 24, 0, 0, 1],
    [1, 3, 0.0307, 0.0621, 0., 17, 0, 0, 1],
    [2, 4, 0.0303, 0.0611, 0., 17, 0, 0, 1],
    [2, 5, 0.0159, 0.0502, 0., 17, 0, 0, 1]
])
