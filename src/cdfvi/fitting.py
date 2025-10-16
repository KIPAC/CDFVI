
import numpy as np

class PolyFit:

    def __init__(self, z_d, z_s, R_vals, R_errs, constrain=False):
        self.z_d = z_d
        self.z_s = z_s
        self.R_vals = R_vals
        self.R_errs = R_errs
        self.constrain = constrain

    @staticmethod
    def chi2Vals(data, errors, model):
        delta = (data - model)/errors
        return delta*delta
        
    @staticmethod
    def modelVals(z_d, z_s, params):
        j_int_poly = np.polynomial.Polynomial(params)
        j_int_s = j_int_poly(z_s)
        j_int_d = j_int_poly(z_d)
        return j_int_d / j_int_s

    @staticmethod
    def modelSpace(z_grid, params):        
        mesh = np.meshgrid(z_grid, z_grid)
        z_d = mesh[0]
        z_s = mesh[1]
        j_int_poly = np.polynomial.Polynomial(params)
        j_int_s = j_int_poly(z_s.ravel()).reshape(z_s.shape)
        j_int_d = j_int_poly(z_d.ravel()).reshape(z_d.shape)
        rat = j_int_d/j_int_s
        return np.where(z_s>z_d, rat, np.nan)
    
    def model(self, params):
        return PolyFit.modelVals(self.z_d, self.z_s, params)

    def chi2(self, params):
        model = self.model(params)
        return PolyFit.chi2Vals(self.R_vals, self.R_errs, model)

    def resid(self, params):
        model = self.model(params)
        resid = self.R_vals - model
        return resid
    
    def __call__(self, params):
        chi2v = self.chi2(params)
        if self.constrain:
            j_int_poly = np.polynomial.Polynomial(params)
            j_poly = j_int_poly.deriv(1)
            delta_h0 = j_poly(0) - 1
            constraint = delta_h0*delta_h0/(0.1*0.1)            
        else:
            constraint = 0.
        return chi2v.sum() + constraint
    
