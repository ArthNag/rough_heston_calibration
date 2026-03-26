import numpy as np

def char_func_standard_heston(a, T, params):
    """
    Standard Heston Characteristic Function (Stable Formulation)
    params = (lambd, rho, nu, theta, V0)
    """
    lambd, rho, nu, theta, V0 = params
    
    # Structural variable d
    d = np.sqrt((rho * nu * 1j * a - lambd)**2 + nu**2 * (1j * a + a**2))
    
    # Numerically stable g
    g = (lambd - rho * nu * 1j * a - d) / (lambd - rho * nu * 1j * a + d)
    
    # Stable exponential coefficients using -d * T
    exp_mdT = np.exp(-d * T)
    
    D = ((lambd - rho * nu * 1j * a - d) / nu**2) * ((1 - exp_mdT) / (1 - g * exp_mdT))
    C = (lambd * theta / nu**2) * ((lambd - rho * nu * 1j * a - d) * T - 2 * np.log((1 - g * exp_mdT) / (1 - g)))
    
    # Note: This returns the variance contribution. 
    # Don't forget to account for the initial log-price and drift (e.g., risk-free rate) elsewhere!
    return np.exp(C + D * V0)
