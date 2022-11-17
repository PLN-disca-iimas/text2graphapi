from joblib import Parallel, delayed


class Utils(object):
    def __init__(self):
        ...

    def joblib_delayed(self, funct, params):
        return delayed(funct)(params)

    def joblib_parallel(self, delayed_funct, n_jobs=-1, backend='loky'):
        return Parallel(
            n_jobs=n_jobs,
            backend=backend
        )(delayed_funct)

    def save_data():
        ...