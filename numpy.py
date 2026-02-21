float32 = "float32"


class NDArray(list):
    def tolist(self):
        return list(self)


def array(values, dtype=None):
    if dtype == float32:
        return NDArray(float(v) for v in values)
    return NDArray(values)

ndarray = NDArray
