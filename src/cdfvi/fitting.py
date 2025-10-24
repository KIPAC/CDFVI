
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from . import utility_functions as funcs



class Chi2Function:
    """Base class to provide a chi^2 function"""

    def __init__(
        self,
        vals: np.ndarray,
        errs: np.ndarray,
    ):
        self.vals = vals
        self.errs = errs
    
    @classmethod
    def chi2Vals(
        cls,
        data: np.ndarray,
        errors: np.ndarray,
        model: np.ndarray
    ) -> np.ndarray:
        delta = (data - model)/errors
        return delta*delta

    def modelSpace(self, z, params, **kwargs: Any):
        return self.modelVals(z, params, **kwargs)
    
    def chi2(self, params):
        model_vals = self.model(params)
        return self.chi2Vals(self.vals, self.errs, model_vals)

    def resid(self, params):
        model_vals = self.model(params)
        resid_vals = self.vals - model_vals
        return resid_vals

    def scaled_resid(self, params):
        resid_vals = self.resid(params)
        return resid_vals/self.errs

    def __call__(self, params):
        chi2v = self.chi2(params)
        return chi2v.sum()

    
class TDPolyChi2(Chi2Function):
    """Class to use time-delay data to fit comoving distance"""
    
    def __init__(
        self,
        z_d: np.ndarray,
        z_s: np.ndarray,
        vals: np.ndarray,
        errs: np.ndarray,
        **kwargs: Any
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z_d = z_d
        self.z_s = z_s
        self.constraint = kwargs.get('constraint', None)
        self.poly_type = kwargs.get('poly_type', np.polynomial.Polynomial)
                
    @classmethod
    def modelVals(
        cls,
        z_d: np.ndarray,
        z_s: np.ndarray,
        params: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        poly_type = kwargs.get('poly_type', np.polynomial.Polynomial)
        j_int_poly = poly_type(params)
        j_int_s = j_int_poly(z_s.ravel()).reshape(z_s.shape)
        j_int_d = j_int_poly(z_d.ravel()).reshape(z_d.shape)
        rat = j_int_d/j_int_s        
        return np.where(z_s>z_d, rat, np.nan)

    def modelSpace(self, z, params, **kwargs):
        mesh = np.meshgrid(z, z)
        z_d = mesh[0]
        z_s = mesh[1]
        return self.modelVals(z_d, z_s, params, **kwargs)
    
    def model(self, params):
        return self.modelVals(self.z_d, self.z_s, params, poly_type=self.poly_type)
    
    def __call__(self, params):
        chi2_val = Chi2Function.__call__(self, params)
        if self.constraint is None:
            return chi2_val
        j_int_poly = self.poly_type(params)
        j_poly = j_int_poly.deriv(1)
        delta_h0 = j_poly(0) - 1
        scaled_delta = delta_h0/self.constraint
        constraint = scaled_delta*scaled_delta
        return chi2_val + constraint
    

class SNPolyChi2(Chi2Function):
    """Class to use SN data to fit comoving distance"""
    
    def __init__(
        self,
        z: np.ndarray,
        vals: np.ndarray,
        errs: np.ndarray,
        **kwargs: Any
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z = z
        self.constraint = kwargs.get('constraint', None)
        self.poly_type = kwargs.get('poly_type', np.polynomial.Polynomial)
                
    @classmethod
    def modelVals(
        cls,
        z: np.ndarray,
        params: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        poly_type = kwargs.get('poly_type', np.polynomial.Polynomial)        
        j_int_poly = poly_type(params)
        return j_int_poly(z)
        
    def model(self, params):
        return SNPolyChi2.modelVals(self.z, params, poly_type=self.poly_type)

    def __call__(self, params):
        chi2_val = Chi2Function.__call__(self, params)
        if self.constraint is None:
            return chi2_val
        j_int_poly = self.poly_type(params)
        j_poly = j_int_poly.deriv(1)
        delta_h0 = j_poly(0) - 1
        scaled_delta = delta_h0/self.constraint
        constraint = scaled_delta*scaled_delta
        return chi2_val + constraint


class CosmoWrapper:

    def __init__(
        self,
        param_names: list[str],
        param_scales: dict[str, float],
        **kwargs,
    ):
        self.param_names = param_names.copy()
        self.param_scales = param_scales.copy()
        self.fixed_params = kwargs.copy()
        self.cosmo = None
        
    def set_params(
        self,
        params: np.ndarray,
    ) -> None:
        assert len(params) == len(self.param_names)
        kw = self.fixed_params.copy()
        for k, v in zip(self.param_names, params):
            if k in self.param_scales:
                scale = self.param_scales[k]
            else:
                scale = 1.
            kw[k] = scale*v
        self.cosmo = FlatLambdaCDM(**kw)
        
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.cosmo.comoving_distance(z).value / 1e3

    
class SNCosmoChi2(Chi2Function):
    """Class to use SN data to fit comoving distance"""
    
    def __init__(
        self,
        z: np.ndarray,
        vals: np.ndarray,
        errs: np.ndarray,
        param_names: list[str],
        param_scales: dict[str, float],
        **kwargs: Any
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z = z
        self.cosmo_wrapper = CosmoWrapper(param_names, param_scales, **kwargs)
                
    @classmethod
    def modelVals(
        cls,
        z: np.ndarray,
        params: np.ndarray,
        cosmo_wrapper: CosmoWrapper,
    ) -> np.ndarray:
        try:
            cosmo_wrapper.set_params(params)
            return cosmo_wrapper(z)
        except ValueError:
            return np.zeros(z.shape)
    
    def model(self, params):
        return SNCosmoChi2.modelVals(self.z, params, cosmo_wrapper=self.cosmo_wrapper)

    def modelSpace(self, z, params, **kwargs: Any):
        return self.modelVals(z, params, self.cosmo_wrapper, **kwargs)

    

class TDCosmoChi2(Chi2Function):
    """Class to use time-delay data to fit comoving distance"""
    
    def __init__(
        self,
        z_d: np.ndarray,
        z_s: np.ndarray,
        vals: np.ndarray,
        errs: np.ndarray,
        param_names: list[str],
        param_scales: dict[str, float],
        **kwargs: Any
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z_d = z_d
        self.z_s = z_s
        self.cosmo_wrapper = CosmoWrapper(param_names, param_scales, **kwargs)
                
    @classmethod
    def modelVals(
        cls,
        z_d: np.ndarray,
        z_s: np.ndarray,
        params: np.ndarray,
        cosmo_wrapper: CosmoWrapper,        
    ) -> np.ndarray:
        cosmo_wrapper.set_params(params)
        chi_s = cosmo_wrapper(z_s.ravel()).reshape(z_s.shape)
        chi_d = cosmo_wrapper(z_d.ravel()).reshape(z_d.shape)
        rat = chi_d/chi_s        
        return np.where(z_s>z_d, rat, np.nan)

    def modelSpace(self, z, params, **kwargs):
        mesh = np.meshgrid(z, z)
        z_d = mesh[0]
        z_s = mesh[1]
        return cls.modelVals(z_d, z_s, params, self.cosmo_wrapper)
    
    def model(self, params):
        return self.modelVals(self.z_d, self.z_s, params, self.cosmo_wrapper)

