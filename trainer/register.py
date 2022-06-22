class RegistrantFactory:
    """
    To register the subclasses based on the name
    """

    @classmethod
    def register(cls, subclass_name):
        def decorator(subclass):
            subclass.subclasses[subclass_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_name, **params):
        if subclass_name not in cls.subclasses:
            raise ValueError("Unknown subclass name {}".format(subclass_name))

        return cls.subclasses[subclass_name](**params)
