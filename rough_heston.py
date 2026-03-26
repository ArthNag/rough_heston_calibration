import sys
import numpy as np
from scipy.integrate import quad
from riccati import solve_riccati

def characteristic_function(a, T, alpha, lambd, rho, nu, theta, V0, n_steps=100):
    t, h, f_vals = solve_riccati(a, T, alpha, lambd, rho, nu, n_steps)

    # g1 = λθ ∫₀ᵀ h(s) ds  (integral of h over the time grid)
    # g2 = h(T)             (final value; equals ∫F dt only when α=1)
    if hasattr(np, 'trapezoid'):
        g1 = theta * lambd * np.trapezoid(h, t)
    else:
        g1 = theta * lambd * np.trapz(h, t)
    g2 = h[-1]

    return np.exp(g1 + V0 * g2)

def price_rough_heston(S0, K, T, r, params, option_type='call'):
    # params = (alpha, lambd, rho, nu, theta, V0)
    k = np.log(S0 / K) + r * T
    
    def integrand(z):
        # shifted frequency for the lewis method 
        phi = characteristic_function(z - 0.5j, T, *params)
        return (np.exp(1j * z * k) * phi / (z**2 + 0.25)).real

    integral, _ = quad(integrand, 0, 100) # integrate up to a high enough cutoff
    
    # CORRECTED: Lewis discount factor is e^{-rT/2} not e^{-rT}?
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
    # 1 creates the fixed grid for the integration variable z
    z_grid = np.linspace(1e-4, z_max, N_z)
    dz = z_grid[1] - z_grid[0]
    
    # 2 pre-compute the characteristic function over the z grid
    # this is the heavy lifting: we only do it ONCE per optimizer step
    # doing it once per strike is too costly and redundant (previous code did this, takes ages to run)
    phi_vals = np.zeros(N_z, dtype=complex)
    for i, z in enumerate(z_grid):
        phi_vals[i] = characteristic_function(z - 0.5j, T, *params)
        
    # 3 apply Lewis method to all strikes using the precomputed phi
    prices = []
    for K in strikes:
        k = np.log(S0 / K) + r * T
        
        # comp the integrand using the pre-computed phi values
        integrand = (np.exp(1j * z_grid * k) * phi_vals / (z_grid**2 + 0.25)).real
        
        # int using the trapezoidal rule
        # use numpy here with "trapz" or "trapezoid"
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

# old mse fct, only for calls or puts, use the joint version now

# def rough_heston_mse(params, market_strikes, market_prices, S0, T, r, option_type='call'):
#     alpha, lambd, rho, nu, theta, V0 = params

#     # domain checks
#     if not (0.5 < alpha < 1.0) or lambd <= 0 or nu <= 0 or V0 <= 0 or theta < 0 or abs(rho) > 1.0:
#         return 1e10

#     try:
#         # Get all model prices in one vectorized call
#         model_prices = price_rough_heston_strip(S0, market_strikes, T, r, params, option_type)
        
#         # Ensure no negative or NaN prices made it through
#         if not np.all(np.isfinite(model_prices)) or np.any(model_prices < 0):
#             return 1e10
            
#     except Exception as e:
#         return 1e10

#     return np.mean((model_prices - np.array(market_prices))**2)

def rough_heston_joint_mse(params, call_strikes, call_prices, put_strikes, put_prices, S0, T, r):
    """
    Joint MSE for calls and puts, using the strip pricing function to avoid redundant Riccati solves.
    This is the main objective function for calibration.
    """
    alpha, lambd, rho, nu, theta, V0 = params

    # 1 strict domain checks (prioritise params to be in the valid space for "rough")
    if not (0.5 < alpha < 1.0) or lambd <= 0 or nu <= 0 or V0 <= 0 or theta < 0 or abs(rho) > 1.0:
        return 1e10

    try:
        # 2 comp model prices for calls
        model_calls = price_rough_heston_strip(S0, call_strikes, T, r, params, 'call')
        
        # 3 comp model prices for puts
        model_puts = price_rough_heston_strip(S0, put_strikes, T, r, params, 'put')
        
        # 4 sanity check on prices (no negative or nan val)
        if not (np.all(np.isfinite(model_calls)) and np.all(np.isfinite(model_puts))):
            return 1e10
            
        # 5 comp the combined mean squared error
        # simply concatenate the errors and get the global mean
        all_errors = np.concatenate([model_calls - call_prices, model_puts - put_prices])
        return np.mean(all_errors**2)

    # if nothing works we just return large error
    except Exception:
        return 1e10

def get_char_func_values(a_grid, T, params, n_steps=1500):
    """
    Computes the characteristic function values for a grid of 'a' values.
    params = (alpha, lambd, rho, nu, theta, V0)
    """
    alpha, lambd, rho, nu, theta, V0 = params
    results = []
    
    for a in a_grid:
        # Solve Riccati for this specific frequency
        t, h, f_vals = solve_riccati(a, T, alpha, lambd, rho, nu, n_steps)
        
        if hasattr(np, 'trapezoid'):
            g1 = theta * lambd * np.trapezoid(h, t)
        else:
            g1 = theta * lambd * np.trapz(h, t)
        g2 = h[-1]

        phi = np.exp(g1 + V0 * g2)
        results.append(phi)
        
    return np.array(results)
