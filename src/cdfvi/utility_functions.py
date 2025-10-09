

def z_to_a(z: np.ndarray) -> np.ndarray:
    """Convert redshift to scale factor"""
    return 1/(1+z)

def a_to_z(a: np.ndarray) -> np.ndarray:
    """Convert scale factor to redshift"""
    return 1/(1-a)


def generalized_sin(x: np.ndarray, K: float=0) -> np.ndarray:
    scaleK = np.sqrt(np.fabs(K))
    if K < 0:
        return np.sin(scaleK*x)/scaleK
    elif K > 0:
        return np.sinh(scaleK*x)/scaleK
    return x


def generalized_cos(x: np.ndarray, K: float=0) -> np.ndarray:
    scaleK = np.sqrt(np.fabs(K))
    if K < 0:
        return np.cos(scaleK*x)/scaleK
    elif K > 0:
        return np.cosh(scaleK*x)/scaleK
    return np.ones(len(x))


def generalized_cot(x: np.ndarray, K: float=0) -> np.ndarray:
    scaleK = np.sqrt(np.fabs(K))
    if K < 0:
        return scaleK/np.tan(scaleK*x)
    elif K > 0:
        return scaleK/np.tanh(scaleK*x)
    return np.ones(len(x))


def R_flat(cosmo, z_d: np.ndarray, z_l: np.ndarray) -> np.ndarray:
    a_d = z_to_a(z_d)
    a_l = z_to_a(z_l)
    a_s = a_d * a_l
    chi_d = cosmo.comoving_radial_distance(a_d.ravel()).reshape(a_d.shape)
    chi_s = cosmo.comoving_radial_distance(a_s.ravel()).reshape(a_s.shape)
    chi_l = chi_s - chi_d
    return (1 + (chi_d/chi_l))/a_s


def R_cosmo(cosmo, z_d: np.ndarray, z_l: np.ndarray) -> np.ndarray:
    a_d = z_to_a(z_d)
    a_l = z_to_a(z_l)
    a_s = a_d * a_l
    dm_d = cosmo.comoving_transverse_distance(a_d.ravel()).reshape(a_d.shape)
    dm_s = cosmo.comoving_transverse_distance(a_s.ravel()).reshape(a_s.shape)
    dm_l = dm_s - dm_d
    return (1 + (dm_d/dm_l))/a_s
