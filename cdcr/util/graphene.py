import functools

from graphene import Field, ObjectType


class ObjectTypeProperty(ObjectType):
    def __init__(self, *args, **kwargs):
        fields = self._meta.fields.items()
        fields = filter(lambda f: isinstance(f, FieldProperty), fields)
        for field in fields:
            field._initialized = False
        super().__init__(*args, **kwargs)
        for field in fields:
            field._initialized = True


class FieldProperty(Field):
    """
    Allow a Field to be a Property.
    FieldProperty takes the same arguments as graphene.Field.

    Usage:

        from graphene import Schema, String
        from newsalyze.util.graphene import ObjectTypeProperty, FieldProperty


        class Query(ObjectTypeProperty):
            field = String()

            @FieldProperty(String)
            def field_copy(self):
                return self.field

            @field_copy.setter
            def field_copy(self, text):
                self.field = text

        schema = Schema(query=Query)

    Emulate PyProperty_Type() similar to Objects/descrobject.c
    See https://docs.python.org/3/howto/descriptor.html#properties"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = None
        self._args = args
        self._kwargs = kwargs
        self.fget = None
        self.fset = None
        self.fdel = None
        self.__doc__ = None

    def __call__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc
        return self

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if value is None and self._initialized is False:
            return

        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        self.fget = fget
        return

    def setter(self, fset):
        self.fset = fset
        return self

    def deleter(self, fdel):
        self.fdel = fdel
        return self


    # def getter(self, fget):
    #     new = type(self)(self._args, self._kwargs)
    #     return new(fget=fget)
    #
    # def setter(self, fset):
    #     new = type(self)(self._args, self._kwargs)
    #     return new(fset=fset)
    #
    # def deleter(self, fdel):
    #     new = type(self)(self._args, self._kwargs)
    #     return new(fdel=fdel)
