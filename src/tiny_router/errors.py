class RouterError(Exception):
    """Base exception for errors callers may handle."""


class DatasetError(RouterError, ValueError):
    """A training or evaluation dataset is invalid."""


class ArtifactError(RouterError, ValueError):
    """A serialized model artifact is invalid or incompatible."""


class ConfigurationError(RouterError, ValueError):
    """Router configuration is invalid."""


class InvalidPromptError(RouterError, ValueError):
    """A prompt cannot be routed because it violates the SDK contract."""


class ProviderError(RouterError):
    """A downstream model invocation failed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ExhaustedError(ProviderError):
    """Every permitted model tier failed or produced an unacceptable answer."""

    def __init__(self, message: str, *, attempts: tuple[object, ...] = ()) -> None:
        super().__init__(message, retryable=False)
        self.attempts = attempts
