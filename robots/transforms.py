import numpy as np

def hom(x, n=1., axis=0):    
    s = [d if i != axis else 1 for i, d in enumerate(x.shape)]
    h = np.full((s), n)
    return np.append(x, h, axis=axis)

def hnorm(x, axis=0, divide=True):
    selh = [slice(None) if i != axis else slice(-1, None) for i in range(len(x.shape))]
    selx = [slice(None) if i != axis else slice(-1) for i in range(len(x.shape))]
    
    if divide:
        h = x[selh]
        return x[selx] / h
    else:
        return np.copy(x[selx])