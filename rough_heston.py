import sys
import numpy as np
from scipy.integrate import quad
from riccati import solve_riccati


def characteristic_function(a, T, alpha, lambd, rho, nu, theta, V0, n_steps=100):
    t, h, f_vals = solve_riccati(a, T, alpha, lambd, rho, nu, n_steps)
    
    # g1 is the standard integral of h
    # numpy 2.4 renamed `trapz` to `trapezoid`; use whichever is available
    if hasattr(np, 'trapezoid'):
        g1 = theta * lambd * np.trapezoid(h, t)
        g2 = np.trapezoid(f_vals, t)
    else:
        g1 = theta * lambd * np.trapz(h, t)
        g2 = np.trapz(f_vals, t)
    
    return np.exp(g1 + V0 * g2)

def price_rough_heston(S0, K, T, r, params, option_type='call'):
    # params = (alpha, lambd, rho, nu, theta, V0)
    k = np.log(S0 / K) + r * T
    
    def integrand(z):
        # shifted frequency for the lewis method 
        phi = characteristic_function(z - 0.5j, T, *params)
        return (np.exp(1j * z * k) * phi / (z**2 + 0.25)).real

    integral, _ = quad(integrand, 0, 100) # integrate up to a high enough cutoff
    
    # CORRECTED: The Lewis discount factor is e^{-rT/2}, not e^{-rT}
    integral_term = (np.sqrt(S0 * K) * np.exp(-0.5 * r * T) / np.pi) * integral
    
    if option_type == 'call':
        price = S0 - integral_term
    elif option_type == 'put':
        price = K * np.exp(-r * T) - integral_term
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return price

# def rough_heston_mse(params, market_strikes, market_prices, S0, T, r, option_type='call'):
#     alpha, lambd, rho, nu, theta, V0 = params

#     # rough domain: bdd must be respected
#     if not (0.5 < alpha < 1.0):
#         return 1e10
#     if lambd <= 0 or nu <= 0 or V0 <= 0 or theta < 0:
#         return 1e10
#     if abs(rho) > 1.0:
#         return 1e10

#     model_prices = []

#     try:
#         for K in market_strikes:
#             p = price_rough_heston(S0, K, T, r, params, option_type)
#             if not np.isfinite(p) or p < 0:
#                 return 1e10
#             model_prices.append(p)
#     except Exception as e:
#         return 1e10

#     mse = np.mean((np.array(model_prices) - np.array(market_prices))**2)
#     return mse

def price_rough_heston_strip(S0, strikes, T, r, params, option_type='call', z_max=50, N_z=150):
    """
    Prices an entire strip of options simultaneously to avoid redundant 
    evaluations of the fractional Riccati equation.
    """
    # 1. Create a fixed grid for the integration variable z
    z_grid = np.linspace(1e-4, z_max, N_z)
    dz = z_grid[1] - z_grid[0]
    
    # 2. Precompute the characteristic function over the z grid
    # This is the heavy lifting, but we only do it ONCE per optimizer step!
    phi_vals = np.zeros(N_z, dtype=complex)
    for i, z in enumerate(z_grid):
        phi_vals[i] = characteristic_function(z - 0.5j, T, *params)
        
    # 3. Apply the Lewis method to all strikes using the precomputed phi
    prices = []
    for K in strikes:
        k = np.log(S0 / K) + r * T
        
        # Calculate the integrand using the precomputed phi values
        integrand = (np.exp(1j * z_grid * k) * phi_vals / (z_grid**2 + 0.25)).real
        
        # Integrate using the trapezoidal rule
        if hasattr(np, 'trapezoid'):
            integral = np.trapezoid(integrand, dx=dz)
        else:
            integral = np.trapz(integrand, dx=dz)
            
        integral_term = (np.sqrt(S0 * K) * np.exp(-0.5 * r * T) / np.pi) * integral
        
        if option_type == 'call':
            prices.append(S0 - integral_term)
        elif option_type == 'put':
            prices.append(K * np.exp(-r * T) - integral_term)
            
    return np.array(prices)

def rough_heston_mse(params, market_strikes, market_prices, S0, T, r, option_type='call'):
    alpha, lambd, rho, nu, theta, V0 = params

    # Domain checks
    if not (0.5 < alpha < 1.0) or lambd <= 0 or nu <= 0 or V0 <= 0 or theta < 0 or abs(rho) > 1.0:
        return 1e10

    try:
        # Get all model prices in one vectorized call
        model_prices = price_rough_heston_strip(S0, market_strikes, T, r, params, option_type)
        
        # Ensure no negative or NaN prices made it through
        if not np.all(np.isfinite(model_prices)) or np.any(model_prices < 0):
            return 1e10
            
    except Exception as e:
        return 1e10

    return np.mean((model_prices - np.array(market_prices))**2)