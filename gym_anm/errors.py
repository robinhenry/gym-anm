"""Error types for :code:`gym-anm`."""


class ANMEnvConfigurationError(Exception):
    """A base class for exceptions relative to the construction of a gym-anm env."""

    pass


class ArgsError(ANMEnvConfigurationError):
    """Raised when one of the environment argument is invalid."""

    pass


class ObsSpaceError(ANMEnvConfigurationError):
    """Raised when the observation space is not properly specified."""

    pass


class ObsNotSupportedError(ObsSpaceError):
    """Raised when an element of the observation vector specified is unsupported."""

    def __init__(self, wanted, allowed):
        super().__init__("Observation type unsupported. Desired {} but we only " "support {}.".format(wanted, allowed))


class UnitsNotSupportedError(ObsSpaceError):
    """Raised when the units specified for the observation vector is unsupported"""

    def __init__(self, wanted, allowed, key):
        super().__init__(
            "Observation unit unsupported. Desired: {} but we only "
            "support {} for observation {}.".format(wanted, allowed, key)
        )


class EnvInitializationError(ANMEnvConfigurationError):
    """Raised when the environment encounters a problem during reset()."""

    pass


class EnvNextVarsError(ANMEnvConfigurationError):
    """Raised when something goes wrong with the :py:func:`next_vars()` function."""
