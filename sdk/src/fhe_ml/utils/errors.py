class FHESDKError(Exception):
    """Base for every error raised by the SDK."""


class ShapeError(FHESDKError, ValueError):
    """Array-like data has the wrong rank, shape, or dtype.

    Subclasses `ValueError` so existing `except ValueError` handlers keep working.
    """


class LayerConfigError(FHESDKError, ValueError):
    """A layer was constructed with invalid arguments.

    Subclasses `ValueError` so existing `except ValueError` handlers keep working.
    """
