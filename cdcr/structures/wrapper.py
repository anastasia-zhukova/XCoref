from typing import Optional


class Wrapper:
    """
    Wrapper which wrappes an attribute and should return its attributes if the wrapper itself
    does not have the attribute.
    """

    WRAPPED_ATTRIBUTE = None
    """
    str: Name of the attribute the wrapper should wrap and point to if an attribute is requested but not found in
    the original object.
    """

    def __getattr__(self, name):
        if hasattr(type(self), name):
            raise AttributeError('An AttributeError was raised by an attribute but ignored. Check your descriptors. '
                                 'It might me necessary to catch all AttributeErrors in a descriptor to find the '
                                 'problem.')
        # This only gets called when no attribute with name was found:
        # Check if ``self.<self.WRAPPED_ATTRIBUTE>`` has it and return it if so.
        wrapped = object.__getattribute__(self, 'WRAPPED_ATTRIBUTE')
        try:
            wrapped = object.__getattribute__(self, wrapped)
        except AttributeError as e:
            message = str(e) + ". Is WRAPPED_ATTRIBUTE correctly set?"
            raise AttributeError(message)

        try:
            return getattr(wrapped, name)
        except AttributeError:
            class_name = type(self).__name__
            raise AttributeError(f"'{class_name}' has no attribute '{name}'")

    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__
