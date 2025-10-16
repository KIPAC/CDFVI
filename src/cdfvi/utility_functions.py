import numpy as np
import pandas
import scipy

def z_to_a(z: np.ndarray) -> np.ndarray:
    """Convert redshift to scale factor"""
    return 1/(1+z)

def a_to_z(a: np.ndarray) -> np.ndarray:
    """Convert scale factor to redshift"""
    return 1/(1-a)


def generalized_sin(x: np.ndarray, K: float=0) -> np.ndarray:
    scale_K = np.sqrt(np.fabs(K))
    if K < 0:
        return np.sin(scale_K*x)/scale_K
    elif K > 0:
        return np.sinh(scale_K*x)/scale_K
    return x


def generalized_cos(x: np.ndarray, K: float=0) -> np.ndarray:
    scale_K = np.sqrt(np.fabs(K))
    if K < 0:
        return np.cos(scale_K*x)/scale_K
    elif K > 0:
        return np.cosh(scale_K*x)/scale_K
    return np.ones(len(x))


def generalized_cot(x: np.ndarray, K: float=0) -> np.ndarray:
    scale_K = np.sqrt(np.fabs(K))
    if K < 0:
        return scale_K/np.tan(scale_K*x)
    elif K > 0:
        return scale_K/np.tanh(scale_K*x)
    return np.ones(len(x))


def chi_ratio(cosmo, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    a_d = z_to_a(z_d)
    a_s = z_to_a(z_s)
    chi_d = cosmo.comoving_radial_distance(a_d.ravel()).reshape(a_d.shape)
    chi_s = cosmo.comoving_radial_distance(a_s.ravel()).reshape(a_s.shape)
    rat = chi_d / chi_s
    return np.where(z_s > z_d, rat, np.nan)


def R_tilde_flat(cosmo, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    a_d = z_to_a(z_d)
    a_s = z_to_a(z_s)
    chi_d = cosmo.comoving_radial_distance(a_d.ravel()).reshape(a_d.shape)
    chi_s = cosmo.comoving_radial_distance(a_s.ravel()).reshape(a_s.shape)
    #rat = chi_s / (a_d * (chi_s - chi_d))
    rat = a_d *(1 - (chi_d / chi_s))
    return np.where(z_s > z_d, rat, np.nan)


def R_tilde_to_R(r_tilde: np.ndarray, a_d: np.ndarray) -> np.ndarray:
    #a_d_times_r_tilde = r_tilde * a_d
    #return np.where(r_tilde > 0, a_d_times_r_tilde / (a_d_times_r_tilde - 1), 0.)
    return np.where(np.isfinite(r_tilde), 1.-(r_tilde/a_d), np.nan)


def R_flat(cosmo, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    a_d = z_to_a(z_d)
    R_tilde = R_tilde_flat(cosmo, z_d, z_s)
    return R_tilde_to_R(R_tilde, a_d)


def h0_over_h(cosmo, a: np.ndarray) -> np.ndarray:
    return 1./cosmo.h_over_h0(a)


def J(cosmo, a: np.ndarray) -> np.ndarray:
    return h0_over_h(cosmo, a)


def J_integral(cosmo, z: np.ndarray) -> np.ndarray:
    a = z_to_a(z)
    j = 1./cosmo.h_over_h0(a)
    z_widths = z[1:] - z[0:-1]    
    return np.cumsum(j[0:-1]) * z_widths


def J_integral_ratio(cosmo, cosmo_ref, z: np.ndarray) -> np.ndarray:
    j_int_ref = J_integral(cosmo_ref, z)
    j_int = J_integral(cosmo, z)
    return j_int / j_int_ref


def simulate_lens_systems(
    n_obj = 1200,
    z_lens_mean = 0.74,
    z_lens_scale = 0.49,
    z_source_mean = 2.31,
    z_source_scale = 0.96,
) -> pandas.DataFrame:

    z_lens_pdf = scipy.stats.lognorm(scale=z_lens_mean, s=z_lens_scale)
    z_source_pdf = scipy.stats.lognorm(scale=z_source_mean, s=z_source_scale)

    z_lens_vals = z_lens_pdf.rvs(n_obj)
    z_source_vals = z_source_pdf.rvs(n_obj)
    mask = np.bitwise_and(
        z_source_vals - z_lens_vals > 0.1,
        z_source_vals < 6.,
    )
    z_lens_vals = z_lens_vals[mask]
    z_source_vals = z_source_vals[mask]

    return pandas.DataFrame(
        dict(
            z_lens=z_lens_vals,
            z_source=z_source_vals,
        )
    )


def function_ratio(function, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    vals_s = function(z_s)
    vals_d = function(z_d)
    return vals_s/vals_d
