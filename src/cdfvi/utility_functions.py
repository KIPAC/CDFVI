from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas
import scipy

if TYPE_CHECKING:
    from pyccl import Cosmology


def z_to_a(z: np.ndarray) -> np.ndarray:
    """Convert redshift to scale factor"""
    return 1 / (1 + z)


def a_to_z(a: np.ndarray) -> np.ndarray:
    """Convert scale factor to redshift"""
    return 1 / (1 - a)


def generalized_sin_curve(x: np.ndarray, curv: float = 0) -> np.ndarray:
    """Return either sin, sinh or x depending on the sign of curvature"""
    scale_curv = np.sqrt(np.fabs(curv))
    if curv < 0:
        return np.sin(scale_curv * x) / scale_curv
    if curv > 0:
        return np.sinh(scale_curv * x) / scale_curv
    return x


def generalized_cos(x: np.ndarray, curv: float = 0) -> np.ndarray:
    """Return either cos, cosh or 1 depending on the sign of curvature"""
    scale_curv = np.sqrt(np.fabs(curv))
    if curv < 0:
        return np.cos(scale_curv * x) / scale_curv
    if curv > 0:
        return np.cosh(scale_curv * x) / scale_curv
    return np.ones(len(x))


def generalized_cot(x: np.ndarray, curv: float = 0) -> np.ndarray:
    """Return either cot, coth or 1/x depending on the sign of curvature"""
    scale_curv = np.sqrt(np.fabs(curv))
    if curv < 0:
        return scale_curv / np.tan(scale_curv * x)
    if curv > 0:
        return scale_curv / np.tanh(scale_curv * x)
    return 1 / x


def chi_ratio(cosmo: Cosmology, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    """Return the ratio of comoving distances"""
    a_d = z_to_a(z_d)
    a_s = z_to_a(z_s)
    chi_d = cosmo.comoving_radial_distance(a_d.ravel()).reshape(a_d.shape)
    chi_s = cosmo.comoving_radial_distance(a_s.ravel()).reshape(a_s.shape)
    rat = chi_d / chi_s
    return np.where(z_s > z_d, rat, np.nan)


def r_tilde_flat(cosmo: Cosmology, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    """Return the predicted r_tilde for a pair of comoving distances

    This uses an approximation for a flat cosmology, and replaces
    the comoving_transverse_distance with the comoving_radial_distance

    r_tilde =  (1 / (1 + z_d)) * (1 - (chi_d / chi_s))
    """
    a_d = z_to_a(z_d)
    a_s = z_to_a(z_s)
    chi_d = cosmo.comoving_radial_distance(a_d.ravel()).reshape(a_d.shape)
    chi_s = cosmo.comoving_radial_distance(a_s.ravel()).reshape(a_s.shape)
    rat = a_d * (1 - (chi_d / chi_s))
    return np.where(z_s > z_d, rat, np.nan)


def r_tilde_to_r(r_tilde: np.ndarray, a_d: np.ndarray) -> np.ndarray:
    """Convert r_tilde to r at a given scale

    r =  1 - (r_tilde / a_d)
    """
    return np.where(np.isfinite(r_tilde), 1.0 - (r_tilde / a_d), np.nan)


def r_flat(cosmo: Cosmology, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    """Compute to observable r for a pair of redshifts

    This uses an approximation for a flat cosmology
    """
    a_d = z_to_a(z_d)
    r_tilde = r_tilde_flat(cosmo, z_d, z_s)
    return r_tilde_to_r(r_tilde, a_d)


def h0_over_h(cosmo: Cosmology, a: np.ndarray) -> np.ndarray:
    """Compute the ratio H_0 / H(z) for a given cosmolgy"""
    return 1.0 / cosmo.h_over_h0(a)


def j_integral_simple(cosmo: Cosmology, z: np.ndarray) -> np.ndarray:
    """Do a simple cumulative sum integraion of H_0/H(z)"""
    a = z_to_a(z)
    j = 1.0 / cosmo.h_over_h0(a)
    z_widths = z[1:] - z[0:-1]
    return np.cumsum(j[0:-1]) * z_widths


def j_integral_ratio(
    cosmo: Cosmology, cosmo_ref: Cosmology, z: np.ndarray
) -> np.ndarray:
    """Compute the ratio of integrals of H(z) for two cosmologies"""
    j_int_ref = j_integral_simple(cosmo_ref, z)
    j_int = j_integral_simple(cosmo, z)
    return j_int / j_int_ref


def simulate_lens_systems(
    n_obj: int = 1200,
    z_lens_mean: float = 0.74,
    z_lens_scale: float = 0.49,
    z_source_mean: float = 2.31,
    z_source_scale: float = 0.96,
) -> pandas.DataFrame:
    """Simluate a set of lens systems"""
    z_lens_pdf = scipy.stats.lognorm(scale=z_lens_mean, s=z_lens_scale)
    z_source_pdf = scipy.stats.lognorm(scale=z_source_mean, s=z_source_scale)

    z_lens_vals = z_lens_pdf.rvs(n_obj)
    z_source_vals = z_source_pdf.rvs(n_obj)
    mask = np.bitwise_and(
        z_source_vals - z_lens_vals > 0.1,
        z_source_vals < 6.0,
    )
    z_lens_vals = z_lens_vals[mask]
    z_source_vals = z_source_vals[mask]

    return pandas.DataFrame(
        dict(
            z_lens=z_lens_vals,
            z_source=z_source_vals,
        )
    )


def function_ratio(function: Callable, z_d: np.ndarray, z_s: np.ndarray) -> np.ndarray:
    """Compute the ratio of a function to two redshifts"""
    vals_s = function(z_s)
    vals_d = function(z_d)
    return vals_s / vals_d
