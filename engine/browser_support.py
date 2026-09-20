"""Only the policy-vector operations used by search.py; no neural emulation."""
class Policy(list):
    def __setitem__(self, index, value):
        if isinstance(index, list):
            values = value if isinstance(value, (list, tuple)) else [value] * len(index)
            if len(index) != len(values):
                raise ValueError('Policy shape mismatch')
            for i, v in zip(index, values):
                super().__setitem__(i, v)
        else:
            super().__setitem__(index, value)


class torch:
    @staticmethod
    def zeros(size):
        return Policy([0.] * size)

    @staticmethod
    def tensor(values):
        return list(values)
