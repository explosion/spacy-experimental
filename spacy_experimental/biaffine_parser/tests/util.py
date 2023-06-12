from thinc.api import Model


def memoize():
    return Model("memoize", memoize_forward, attrs={"X": [], "dY": []})


def memoize_forward(model: Model, X, is_train):
    model.attrs["X"].append(X)

    def backprop(dY):
        model.attrs["dY"].append(dY)
        return dY

    return X, backprop
