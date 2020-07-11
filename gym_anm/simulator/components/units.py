import numpy as np
import numbers

from gym_anm.simulator.components.errors import UnitConversionError

def convert_units(x, old, new, baseMVA=1, basekV=1):
    """
    Converts electrical quantities between different units.

    Parameters
    ----------
    x : Any
        The quantities to convert from `old` to `new` units.
    old : str
        The old units in {`pu`, `MW`, `MVAr`, `kV`, `kA`, `rad`, `degree`}.
    new : str
        The new units (see `old`).
    baseMVA : int
        The base power of the system (MVA).
    basekV : int
        The base voltage of the bus (kV).

    Returns
    -------
    Any
        The input quantities `x` converted to the new units.
    """

    mappings = {('pu', 'MW'): baseMVA,
                ('pu', 'MVAr'): baseMVA,
                ('pu', 'kV'): basekV,
                ('pu', 'kA'): baseMVA / basekV,
                ('rad', 'degree'): 180 / np.pi}

    if (old, new) in mappings.keys():
        factor = mappings[(old, new)]
    elif (new, old) in mappings.keys():
        factor = mappings[(new, old)]
    else:
        raise UnitConversionError(old, new)

    if isinstance(x, numbers.Number):
        return x * factor
    elif isinstance(x, list):
        return [y * factor for y in x]
    elif isinstance(x, dict):
        return {(k, v * factor) for k, v in x}
    elif isinstance(x, np.ndarray):
        return x * factor
    else:
        raise UnitConversionError(old, new)
