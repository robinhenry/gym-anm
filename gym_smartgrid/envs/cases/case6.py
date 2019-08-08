import numpy as np

##### CASE FILE DESCRIPTION #####

### 1. Bus data:
# BUS_I: bus number (positive integer).
# TYPE: bus type (1 = PQ, 2 = PV, 3 = slack).
# PD: real power demand (MW).
# QD: reactive power demand (MVAr).
# GS: shunt conductance (MW demanded at V = 1.0 p.u.).
# BS: shunt susceptance (MVAr injected at V = 1.0 p.i.).
# BUS_AREA: area number (positive integer).
# VM: voltage magnitude (p.u.).
# VA: voltage angle (degrees).
# BASE_KV: base voltage (kV).
# ZONE: loss zone (positive integer).
# VMAX: maximum voltage magnitude (p.u.).
# VMIN: minimum voltage magnitude (p.u.).

# Note: GS, BS, BUS_AREA, ZONE fields are not used.

### 2. Generator data:
# GEN_BUS: bus number.
# PG: real power output (MW).
# QG: reactive power output (MVAr).
# QMAX: maximum reactive power output (MVAr).
# QMIN: minimum reactive power output (MVAr).
# VG: voltage magnitude setpoint (p.u.).
# MBASE: total MVA base of machine, defaults to baseMVA.
# GEN_STATUS: machine status (> 0 = in service).
# PMAX: maximum real power output (MW).
# PMIN: minimum real power output (MW).
# PC1: lower real power output of PQ capability curve (MW).
# PC2: upper real power output of PQ capability curve (MW).
# QC1MIN: minimum reactive power output at PC1 (MVAr).
# QC1MAX: maximum reactive power output at PC1 (MVAr).
# QC2MIN: minimum reactive power output at PC2 (MVAr).
# QC2MAX: maximum reactive power output at PC2 (MVAr).
# VRE_TYPE: type of resource: 0 -> slack (or non-VRE generator), -1 -> load, 1 -> wind, 2 -> solar.

### 3. Branch data.
# F_BUS: "from" bus number.
# T_BUS: "to" bus number.
# BR_R: resistance (p.u.).
# BR_X: reactance (p.u.).
# BR_B: total line charging susceptance (p.u.).
# RATE_A: MVA rating A (long term rating), set to 0 for unlimited.
# RATE_B: MVA rating B (medium term rating), set to 0 for unlimited.
# RATE_C: MVA rating C (short term rating), set to 0 for unlimited.
# TAP: transformer off-nominal turns ratio. If non-zero, taps located at
# 'from' bus and impedance at 'to' bus (see pi-model); if zero, indicating
# no-transformer (i.e. a simple transmission line).
# SHIFT: transformer phase shit angle (degrees), positive => delay.
# BR_STATUS: branch status, 1 = in service, 0 = out-of-service.
# ANGMIN: minimum angle difference (theta_f - theta_t) (degrees).
# ANGMAX: maximum angle difference (theta_f - theta_t) (degrees).
# PF: real power injected at 'from' bus end (MW).
# QF: reactive power injected at 'from' bus end (MVAr).
# PT: real power injected at 'to' bus end (MW).
# QT: reactive power injected at 'to' bus end (MVAr).

# Note: PF - QT fields are not used.

### 4. Storage data.
# BUS_I: bus number.
# SOC_MAX: maximum state of charge (MWh).
# EFF: round-trip efficiency coefficient.
# PMAX: maximum magnitude of real power output (charge and discharge) (MW).
# QMAX: maximum magnitude reactive power output (charge and discharge) (MVAr).
# PC1: lower real power output of PQ capability curve (MW).
# PC2: upper real power output of PQ capability curve (MW).
# QC1MIN: minimum reactive power output at PC1 (MVAr).
# QC1MAX: maximum reactive power output at PC1 (MVAr).
# QC2MIN: minimum reactive power output at PC2 (MVAr).
# QC2MAX: maximum reactive power output at PC2 (MVAr).


def load():

    case = {"version": "ANM"}

    case["baseMVA"] = 100.0

    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    case["bus"] = np.array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9],
        [2, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9],
        [3, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9],
        [4, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 13.8, 1, 1.1, 0.9]
    ])

    # gen_bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin Pc1 Pc2 Qc1min Qc1max
    # Qc2min Qc2max vre_type
    case["gen"] = np.array([
        [1,  0., 0., 100, -100, 1.04, 100, 1, 100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
        [4, -1., -0.267,   0,    0,  0.0, 100, 1, 0., -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1],
        [4, 1., 0.1, 5, -5, 0.0, 100, 1, 20.0, 0.0, 7.5, 20.0, -5., 5., -2., 2., 2],
        [5, -1., -0.254,   0,    0,  0.0, 100, 1, .0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1],
        [5, 1., 0.1, 5, -5, 0.0, 100, 1, 20.0, 0.0, 7.5, 20.0, -5., 5., -2., 2., 1],
        [6, -1., -0.254, 0, 0, 0.0, 100, 1, .0, -100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1],
    ])

    # fbus tbus r x b rateA rateB rateC ratio angle status angmin angmax
    case["branch"] = np.array([
        [1, 2, 0.035, 0.02, 0., 14, 14, 14, 0, 0, 1, -360.0, 360.0],
        [2, 3, 0.5, 0.4, 0., 14, 14, 14, 0, 0, 1, -360.0, 360.0],
        [2, 4, 0.5, 0.4, 0.,  7,  7,  7, 0, 0, 1, -360.0, 360.0],
        [3, 5, 0.5, 0.4, 0.,  7,  7,  7, 0, 0, 1, -360.0, 360.0],
        [3, 6, 0.5, 0.4, 0., 7, 7, 7, 0, 0, 1, -360.0, 360.0]
    ])

    # bus_i soc_max round_trip_eff_coef Pmax Qmax Qmin Pc1 Pc2 Qc1min Qc1max
    # Qc2min Qc2max
    case["storage"] = np.array([
        [6, 50, 0.9, 5, 3, 5, 7, -2.94, 2.94, -1.69, 1.69],
    ])

    return case