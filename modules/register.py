class RegistrantFactory:
    subclasses = {}

    @classmethod
    def register(cls, message_type):
        def decorator(subclass):
            cls.subclasses[message_type] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, subclass_type, **params):
        if subclass_type not in cls.subclasses:
            raise ValueError("Bad subclass type {}".format(subclass_type))

        return cls.subclasses[subclass_type](params)
