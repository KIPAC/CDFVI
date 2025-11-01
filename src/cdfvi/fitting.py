from typing import Any, TypeAlias

import numpy as np
from astropy.cosmology import FlatLambdaCDM

ParamSet: TypeAlias = np.ndarray | list | tuple


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
    def chi2_vals(
        cls, data: np.ndarray, errors: np.ndarray, model: np.ndarray
    ) -> np.ndarray:
        """Return naive chi**2 values

        This ignores correlations

        chi_i**2 = ((data_i - model_i) / errors_i)**2
        """

        chi_values = (data - model) / errors
        return chi_values * chi_values

    def chi2(self, params: ParamSet) -> np.ndarray:
        """Return naive chi**2 values

        This ignores correlations

        chi**2 = Sum_i ((data_i - model_i) / errors_i)**2
        """
        model_vals = self.model(params)
        return self.chi2_vals(self.vals, self.errs, model_vals)

    def model(self, params: ParamSet) -> np.ndarray:
        """Compute the model for values in the data"""
        raise NotImplementedError()

    def resid(self, params: ParamSet) -> np.ndarray:
        """Return the residuals

        resid_i = data_i - model_i
        """
        model_vals = self.model(params)
        resid_vals = self.vals - model_vals
        return resid_vals

    def scaled_resid(self, params: ParamSet) -> np.ndarray:
        """Return the scaled residuals

        scaled_resid_i = (data_i - model_i) / errors_i
        """
        resid_vals = self.resid(params)
        return resid_vals / self.errs

    def __call__(self, params: ParamSet) -> float:
        """Compute the chi**2

        Sub-classes can override this to add a constraint term
        """
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
        **kwargs: Any,
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z_d = z_d
        self.z_s = z_s
        self.constraint = kwargs.get("constraint", None)
        self.poly_type = kwargs.get("poly_type", np.polynomial.Polynomial)

    @classmethod
    def _model_vals(
        cls, z_d: np.ndarray, z_s: np.ndarray, params: ParamSet, **kwargs: Any
    ) -> np.ndarray:
        """Compute the model prediction for a set of pairs of z values"""
        poly_type = kwargs.get("poly_type", np.polynomial.Polynomial)
        j_int_poly = poly_type(params)
        j_int_s = j_int_poly(z_s.ravel()).reshape(z_s.shape)
        j_int_d = j_int_poly(z_d.ravel()).reshape(z_d.shape)
        rat = j_int_d / j_int_s
        return np.where(z_s > z_d, rat, np.nan)

    def model_space(self, z: np.ndarray, params: ParamSet, **kwargs: Any) -> np.ndarray:
        """Compute the model a grid of z-values"""
        mesh = np.meshgrid(z, z)
        z_d = mesh[0]
        z_s = mesh[1]
        return self._model_vals(z_d, z_s, params, **kwargs)

    def model(self, params: ParamSet) -> np.ndarray:
        """Compute the model for the pairs of z-values from the data"""
        return self._model_vals(self.z_d, self.z_s, params, poly_type=self.poly_type)

    def __call__(self, params: ParamSet) -> float:
        """Compute the chi**2 including the constraint term"""
        chi2_val = Chi2Function.__call__(self, params)
        if self.constraint is None:
            return chi2_val
        j_int_poly = self.poly_type(params)
        j_poly = j_int_poly.deriv(1)
        delta_h0 = j_poly(0) - 1
        scaled_delta = delta_h0 / self.constraint
        constraint = scaled_delta * scaled_delta
        return chi2_val + constraint


class SNPolyChi2(Chi2Function):
    """Class to use SN data to fit comoving distance"""

    def __init__(
        self, z: np.ndarray, vals: np.ndarray, errs: np.ndarray, **kwargs: Any
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z = z
        self.constraint = kwargs.get("constraint", None)
        self.poly_type = kwargs.get("poly_type", np.polynomial.Polynomial)

    @classmethod
    def _model_vals(
        cls,
        z: np.ndarray,
        params: ParamSet,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the model prediction for a set of z values"""
        poly_type = kwargs.get("poly_type", np.polynomial.Polynomial)
        j_int_poly = poly_type(params)
        return j_int_poly(z)

    def model(self, params: ParamSet) -> np.ndarray:
        """Compute the model for the z-values from the data"""
        return SNPolyChi2._model_vals(self.z, params, poly_type=self.poly_type)

    def model_space(self, z: np.ndarray, params: ParamSet, **kwargs: Any) -> np.ndarray:
        """Compute the model a grid of z-values"""
        return self._model_vals(z, params, **kwargs)

    def __call__(self, params: ParamSet) -> float:
        """Compute the chi**2 including the constraint term"""
        chi2_val = Chi2Function.__call__(self, params)
        if self.constraint is None:
            return chi2_val
        j_int_poly = self.poly_type(params)
        j_poly = j_int_poly.deriv(1)
        delta_h0 = j_poly(0) - 1
        scaled_delta = delta_h0 / self.constraint
        constraint = scaled_delta * scaled_delta
        return chi2_val + constraint


class CosmoWrapper:
    """Wrapper for FlatLambdaCDM object"""

    def __init__(
        self,
        param_names: list[str],
        param_scales: dict[str, float],
        **kwargs: Any,
    ):
        self.param_names = param_names.copy()
        self.param_scales = param_scales.copy()
        self.fixed_params = kwargs.copy()
        self.cosmo: FlatLambdaCDM | None = None

    def set_params(
        self,
        params: ParamSet,
        **kwargs: Any,
    ) -> None:
        """Build a new FlatLambdaCDM object with the input parameters"""
        assert len(params) == len(self.param_names)
        kw = self.fixed_params.copy()
        kw.update(**kwargs)
        for k, v in zip(self.param_names, params):
            if k in self.param_scales:
                scale = self.param_scales[k]
            else:
                scale = 1.0
            kw[k] = scale * v
        self.cosmo = FlatLambdaCDM(**kw)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        assert self.cosmo is not None
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
        **kwargs: Any,
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z = z
        self.cosmo_wrapper = CosmoWrapper(param_names, param_scales, **kwargs)

    @classmethod
    def _model_vals(
        cls,
        z: np.ndarray,
        params: ParamSet,
        cosmo_wrapper: CosmoWrapper,
    ) -> np.ndarray:
        """Compute the model prediction for a set of z values"""
        try:
            cosmo_wrapper.set_params(params)
            return cosmo_wrapper(z)
        except ValueError:
            return np.zeros(z.shape)

    def model(self, params: ParamSet) -> np.ndarray:
        """Compute the model for the z-values from the data"""
        return SNCosmoChi2._model_vals(self.z, params, cosmo_wrapper=self.cosmo_wrapper)

    def model_space(self, z: np.ndarray, params: ParamSet, **kwargs: Any) -> np.ndarray:
        """Compute the model a grid of z-values"""
        return self._model_vals(z, params, self.cosmo_wrapper, **kwargs)


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
        **kwargs: Any,
    ):
        Chi2Function.__init__(self, vals, errs)
        self.z_d = z_d
        self.z_s = z_s
        self.cosmo_wrapper = CosmoWrapper(param_names, param_scales, **kwargs)

    @classmethod
    def _model_vals(
        cls,
        z_d: np.ndarray,
        z_s: np.ndarray,
        params: ParamSet,
        cosmo_wrapper: CosmoWrapper,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute the model prediction for a set of pairs of z values"""
        cosmo_wrapper.set_params(params, **kwargs)
        chi_s = cosmo_wrapper(z_s.ravel()).reshape(z_s.shape)
        chi_d = cosmo_wrapper(z_d.ravel()).reshape(z_d.shape)
        rat = chi_d / chi_s
        return np.where(z_s > z_d, rat, np.nan)

    def model_space(self, z: np.ndarray, params: ParamSet, **kwargs: Any) -> np.ndarray:
        """Compute the model a grid of z-values"""
        mesh = np.meshgrid(z, z)
        z_d = mesh[0]
        z_s = mesh[1]
        return self._model_vals(z_d, z_s, params, self.cosmo_wrapper, **kwargs)

    def model(self, params: ParamSet) -> np.ndarray:
        """Compute the model for the z-values from the data"""
        return self._model_vals(self.z_d, self.z_s, params, self.cosmo_wrapper)
