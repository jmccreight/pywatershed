from collections.abc import MutableMapping


class DictAccess(MutableMapping):
    """
    Mapping that works like both a dict and a mutable object, i.e.
    d = D(foo='bar')
    and
    d.foo returns 'bar'
    and can have custome __getitem__
    https://stackoverflow.com/questions/21361106/how-would-i-implement-a-dict-with-abstract-base-classes-in-python
    """

    # ``__init__`` method required to create instance from class.
    def __init__(self, *args, **kwargs):
        """Use the object dict"""
        self.__dict__.update(*args, **kwargs)

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        if isinstance(key, list):
            return DictAccess(**{kk: self.__dict__[kk] for kk in key})
        else:
            return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self):
        """echoes class, id, & reproducible representation in the REPL"""
        return f"{self.__dict__}"


# dd = DictAccess()
# print(dd)
# dd["foo"] = 0
# dd["bar"] = 1
# dd["baz"] = 2
# print(dd)
# dd["bar"]
# zz = dd[["foo", "bar"]]
