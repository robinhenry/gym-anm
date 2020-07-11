class InputNetworkFileError(Exception):
    """Base class for exceptions relative to the network input dictionary."""
    def __init__(self, message=""):
        super().__init__(message)

class BaseMVAError(InputNetworkFileError):
    """Raised when the baseMVA for the network is <= 0."""
    def __init__(self):
        super().__init__("The network baseMVA should be > 0.")

class BranchSpecError(InputNetworkFileError):
    """Raised when the specs for a branch are not correctly specified."""
    pass

class BusSpecError(InputNetworkFileError):
    """Raised when the specs for a bus are not correctly specified."""
    pass

class DeviceSpecError(InputNetworkFileError):
    """Raised when the specs for a device are not correctly specified."""
    pass

class GenSpecError(DeviceSpecError):
    """Raised when the specs for a generator are not correctly specified."""
    pass

class LoadSpecError(DeviceSpecError):
    """Raised when the specs for a load are not correctly specified."""
    pass

class StorageSpecError(DeviceSpecError):
    """Raised when the specs for a storage unit are not correctly specified."""
    pass

class PFEError(Exception):
    """Raised when no solution to the network equations is found."""
    pass

class UnitConversionError(Exception):
    """Raised when a convertion between different units failed."""
    def __init__(self, old, new):
        message = 'Cannot convert from %s units to %s units' % (old, new)
        super().__init__(message)
