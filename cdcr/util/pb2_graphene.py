from contextlib import suppress
from typing import Callable, Tuple, Any, Type, Union, TypeVar

import graphene
from google.protobuf.reflection import GeneratedProtocolMessageType
from graphene import ObjectType, String, Field, Enum, List, Scalar
from google.protobuf.pyext._message import EnumDescriptor, FieldDescriptor, FileDescriptor
import google.protobuf.type_pb2 as type_pb2

TYPES = type_pb2.Type.fields.DESCRIPTOR
BaseDescriptor = type(TYPES).__base__

TYPE_MAPPING = {
    TYPES.TYPE_DOUBLE: graphene.Float,
    TYPES.TYPE_FLOAT: graphene.Float,
    TYPES.TYPE_INT64: graphene.Int,
    TYPES.TYPE_UINT64: graphene.Int,
    TYPES.TYPE_INT32: graphene.Int,
    TYPES.TYPE_FIXED64: graphene.Int,
    TYPES.TYPE_FIXED32: graphene.Int,
    TYPES.TYPE_BOOL: graphene.Boolean,
    TYPES.TYPE_STRING: graphene.String,
    # TYPES.TYPE_GROUP: '',  # Deprecated thus not supported.
    # Parsed by hand: TYPES.TYPE_MESSAGE: '',
    # Parsed by hand: TYPES.TYPE_BYTES: '',
    TYPES.TYPE_UINT32: graphene.Int,
    # Parsed by hand: TYPES.TYPE_ENUM: '',
    TYPES.TYPE_SFIXED32: graphene.Int,
    TYPES.TYPE_SFIXED64: graphene.Int,
    TYPES.TYPE_SINT32: graphene.Int,
    TYPES.TYPE_SINT64: graphene.Int,
}
"""Dict[int, Scalar]: Mapping of protobuf2 type to the matching native graphene scalar."""


# noinspection PyUnreachableCode
class Protobuf2Graphene:
    def __init__(self,
                 descriptor: FileDescriptor,
                 name_func: Callable = lambda x: x.name,
                 bases: Tuple[Type[Any], ...] = (ObjectType,)):
        """
        Create nested graphene objects from Protobuf 2 descriptors.

        Example:

            from some.package_pb2 import DESCRIPTOR, SomeClass
            from graphene import Schema, Field, ObjectType

            protobuf2graphene = Protobuf2Graphene(DESCRIPTOR)
            SomeClassGraphQL = protobuf2graphene(SomeClass)

            class Query(ObjectType):
                some_variable = Field(SomeClassGraphQL)

            schema = Schema(query=Query)

        Args:
            descriptor: The descriptor describing a complete protobuf 2 file.
            name_func: Function for creating the class names.
                Takes one argument (BaseDescriptor): The descriptor to add to
                the schema.
            bases: Class bases for fields of type TYPE_MESSAGE.
        """
        self.name_func = name_func
        self.bases = bases
        self._class_by_full_name = {}

        self._initialize_classes(descriptor)

    def __call__(self, pb: Union[GeneratedProtocolMessageType, BaseDescriptor]) -> Union[ObjectType, Scalar]:
        """
        Get matching schema for a protobuf 2 class / descriptor.

        Args:
            pb: Protobuf 2 class or protobuf 2 descriptor to get a matching schema
                for.

        Returns:
            The matching schema or scalar.
        """
        with suppress(AttributeError):
            pb = pb.DESCRIPTOR
        return self._class_by_full_name[pb.full_name]

    def _initialize_classes(self, descriptor: FileDescriptor):
        enums = descriptor.enum_types_by_name
        messages = descriptor.message_types_by_name
        for enum in enums:
            self._create_enum(enums[enum])
        for message in messages:
            self._initialize_class(messages[message])

    def _get_create(self, field) -> Callable:
        """
        Get a the schema for a field.
        Create if not exists.

        Args:
            field: Field to get the schema for.

        Returns:
            Function taking no arguments to getting the schema.
            Required because of circular dependencies.
            See https://github.com/graphql-python/graphene/issues/110#issuecomment-366515268
        """
        def __get_create():
            with suppress(KeyError):
                return self._class_by_full_name[field.full_name]
            if not hasattr(field, "type") or field.type is field.TYPE_MESSAGE:
                return self._initialize_class(field)
        return __get_create

    def _initialize_class(self, descriptor):
        variables = {}
        fields = descriptor.fields_by_name

        # Add all variables (fields) to the schema
        for field_name in fields:
            field = fields[field_name]

            if self._is_enum(field):
                variables[field_name] = Field(self._class_by_full_name[field.enum_type.full_name])
                continue

            if self._is_list(field):
                with suppress(AttributeError, KeyError):
                    variables[field_name] = List(TYPE_MAPPING[field.message_type.type])
                    continue
                variables[field_name] = List(self._get_create(field.message_type))
                continue

            with suppress(AttributeError, KeyError):
                variables[field_name] = TYPE_MAPPING[field.type]()
                continue

            variables[field_name] = Field(self._get_create(field))

        class_ = type(self.name_func(descriptor), self.bases, variables)
        self._class_by_full_name[descriptor.full_name] = class_
        return class_

    @staticmethod
    def _is_list(descriptor: BaseDescriptor):
        with suppress(AttributeError):
            if descriptor.type is not descriptor.TYPE_MESSAGE:
                return False
        return isinstance(descriptor, FieldDescriptor)

    @staticmethod
    def _is_enum(descriptor: BaseDescriptor):
        with suppress(AttributeError):
            if descriptor.type is descriptor.TYPE_ENUM:
                return True
        return isinstance(descriptor, EnumDescriptor)

    def _create_enum(self, descriptor: EnumDescriptor):
        desc_values = descriptor.values_by_name
        enum = Enum(self.name_func(descriptor), [
            (key, desc_values[key].number)
            for key in desc_values
        ])
        self._class_by_full_name[descriptor.full_name] = enum
        return enum
