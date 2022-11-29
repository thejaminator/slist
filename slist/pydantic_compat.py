from typing import Sequence, Generic, TYPE_CHECKING, cast

from slist import Slist, A

if TYPE_CHECKING:
    SlistPydantic = Slist
else:

    class SlistPydanticImpl(Generic[A]):
        """
        To use:
        >>> from pydantic import BaseModel
        >>>class MyModel(BaseModel):
        >>>    my_slist: SlistPydantic[int] # rather than Slist[int]. Otherwise pydantic turns it into a normal list
        >>>my_instance = MyModel(my_slist=Slist([1,2,3]))
        >>>assert my_instance.my_slist.first_option == 1
        """

        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v, field):  # field: ModelField
            subfield = field.sub_fields[0]  # e.g. the int type in Slist[int]
            if not isinstance(v, Sequence):
                raise TypeError(f"Sequence required to instantiate a Slist, got {v} of type {type(v)}")
            validated_values = []
            for idx, item in enumerate(v):
                valid_value, error = subfield.validate(item, {}, loc=str(idx))
                if error is not None:
                    raise ValueError(f"Error validating {item}, Error: {error}")

                validated_values.append(valid_value)
            return Slist(validated_values)

    # Pycharm doesn't check if TYPE_CHECKING so we do this hack
    SlistPydantic = cast(Slist, SlistPydanticImpl)
