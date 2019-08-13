import numpy as np

##### CASE FILE DESCRIPTION #####

### Metadata ###
# baseMVA: base power of the system (MVA).
# basekV: base voltage of the network, assuming single zone (kV).

### 1. Bus data:
# BUS_I: bus number (0-indexing).
# BUS_TYPE: bus type (1 = PQ, 2 = PV, 3 = slack).
# GS: shunt conductance (MW demanded at V = 1.0 p.u.).
# BS: shunt susceptance (MVAr injected at V = 1.0 p.i.).
# VMAX: maximum voltage magnitude (p.u.).
# VMIN: minimum voltage magnitude (p.u.).

# Note: GS, BS fields are not used.

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
# RATE_A: MVA rating A (long term rating), set to 0 for unlimited.
# TAP: transformer off-nominal turns ratio. If non-zero, taps located at
# 'from' bus and impedance at 'to' bus (see pi-model); if zero, indicating
# no-transformer (i.e. a simple transmission line).
# SHIFT: transformer phase shit angle (degrees), positive => delay.
# BR_STATUS: branch status, 1 = in service, 0 = out-of-service.
# ANGMIN: minimum angle difference (theta_f - theta_t) (degrees).
# ANGMAX: maximum angle difference (theta_f - theta_t) (degrees).

network = {"version": "ANM"}

network["baseMVA"] = 100.0
network["basekV"] = 15.0

network["bus"] = np.array([
    [0, 3, 0, 0, 1.04, 1.04],
    [1, 1, 0, 0, 1.1, 0.9],
    [2, 1, 0, 0, 1.1, 0.9],
    [3, 1, 0, 0, 1.1, 0.9],
    [4, 1, 0, 0, 1.1, 0.9],
    [5, 1, 0, 0, 1.1, 0.9]
])

network["device"] = np.array([
    [0,  0,   0., 100, -100,  1, 100, -100,   0,  0,     0,    0,     0,    0,  0,   0],
    [3, -1, 0.267,  0,    0,  1,   0,  -10,   0,  0,     0,    0,     0,    0,  0,   0],
    [3,  3,  0.1,   5,   -5,  1,  20,    0, 7.5, 30,    -5,    5,    -2,    2,  0,   0],
    [4, -1, 0.254,   0,    0,  1,   0,  -50,   0,  0,     0,    0,     0,    0,  0,   0],
    [4,  2,  0.1,   5,   -5,  1,  20,    0, 7.5, 30,    -5,    5,    -2,    2,  0,   0],
    [5, -1, 0.254,   0,    0,  1,   0,  -20,   0,  0,     0,    0,     0,    0,  0,   0],
    [5,  4,    0,   3,   -3,  1,   5,   -5,   5,  7, -2.94, 2.94, -1.69, 1.69, 50, 0.9]
])

network["branch"] = np.array([
    [0, 1, None, None, 0., None, None, None, 1, -360.0, 360.0],   # HV/MV transformer
    [1, 2, 0.5, 0.5, 0.,  None, 0, 0, 1, -360.0, 360.0],          # low current rating
    [1, 3, 0.5, 0.5, 0., 100., 0, 0, 1, -360.0, 360.0],           # high current rating, average resistance
    [2, 4, 0.5, 0.5, 0., 100., 0, 0, 1, -360.0, 360.0],           # high current rating, average resistance
    [2, 5, 1.2479, 1.2479, 0., None, 0, 0, 1, -360.0, 360.0]      # high resistance
])
