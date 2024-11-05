"""
This module provides a registry for HFUNet models.
"""

HFUNET_MODEL_REGISTRY = {}


def register_model(name):
    """
    Registers a model class with the given name.

    Args:
        name (str): The name of the model to register.

    Returns:
        function: The decorator function to register the model class.
    """
    def register_model_cls(cls):
        if name in HFUNET_MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        HFUNET_MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls
