import numpy as np
from scipy.special import gamma

# def F_riccati(a, x, lambd, rho, nu):
#     # paper's equation (24) and (422)
#     term1 = 0.5 * (-a**2 - 1j * a)
#     term2 = lambd * (1j * a * rho * nu - 1) * x
#     term3 = 0.5 * (lambd * nu)**2 * x**2
#     return term1 + term2 + term3

# def solve_riccati(a, T, alpha, lambd, rho, nu, n_steps=300):
#     dt = T / n_steps
#     t = np.linspace(0, T, n_steps + 1)
#     h = np.zeros(n_steps + 1, dtype=complex)
#     f_vals = np.zeros(n_steps + 1, dtype=complex)

#     # Pre-calculate weights for speed
#     # Weights for the predictor (Riemann sum)
#     def get_b(k):
#         return ((np.arange(k + 1) + 1)**alpha - (np.arange(k + 1))**alpha)

#     # weights for the corrector (using trapezoidal rule)
#     def get_a(k):
#         j = np.arange(k + 1)
#         a_weights = np.zeros(k + 2)
#         a_weights[0] = (k**(alpha + 1) - (k - alpha) * (k + 1)**alpha)
#         if k > 0:
#             a_weights[1:k+1] = ((k - j[1:] + 2)**(alpha + 1) + (k - j[1:])**(alpha + 1) - 2*(k - j[1:] + 1)**(alpha + 1))
#         a_weights[k+1] = 1
#         return a_weights * (dt**alpha / gamma(alpha + 2))

#     for k in range(n_steps):
#         # Predictor step
#         b = get_b(k)
#         h_predict = (dt**alpha / gamma(alpha + 1)) * np.sum(b * f_vals[:k+1][::-1])
        
#         # Corrector step
#         weights_a = get_a(k)
#         f_predict = F_riccati(a, h_predict, lambd, rho, nu)
#         h[k+1] = np.sum(weights_a[:-1] * f_vals[:k+1]) + weights_a[-1] * f_predict
#         f_vals[k+1] = F_riccati(a, h[k+1], lambd, rho, nu)

#     return t, h, f_vals

def F_riccati(a, x, lambd, rho, nu):
    term1 = 0.5 * (-a**2 - 1j * a)
    term2 = lambd * (1j * a * rho * nu - 1) * x
    term3 = 0.5 * (lambd * nu)**2 * x**2
    return term1 + term2 + term3

def solve_riccati(a, T, alpha, lambd, rho, nu, n_steps=100):
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    h = np.zeros(n_steps + 1, dtype=complex)
    f_vals = np.zeros(n_steps + 1, dtype=complex)

    # b_diff[j] = (j+1)^alpha - j^alpha
    delta_alpha = np.arange(n_steps + 2)**alpha
    b_diff = delta_alpha[1:] - delta_alpha[:-1]
    b_coef = dt**alpha / gamma(alpha + 1)
    
    # weights base array
    a_coef = dt**alpha / gamma(alpha + 2)
    
    for k in range(n_steps):
        # predictor step
        b_w = b_coef * b_diff[k::-1] 
        h_predict = np.sum(b_w * f_vals[:k+1])
        
        # corrector step
        f_predict = F_riccati(a, h_predict, lambd, rho, nu)
        
        # Calculate a weights for this specific step 'k'
        a_w = np.zeros(k + 2)
        a_w[0] = a_coef * (k**(alpha + 1) - (k - alpha) * (k + 1)**alpha)
        if k > 0:
            j = np.arange(1, k + 1)
            a_w[1:k+1] = a_coef * ((k - j + 2)**(alpha + 1) + (k - j)**(alpha + 1) - 2*(k - j + 1)**(alpha + 1))
        a_w[k+1] = a_coef
        
        # Current f_vals slice plus the predicted f value
        h[k+1] = np.sum(a_w[:-1] * f_vals[:k+1]) + a_w[-1] * f_predict
        f_vals[k+1] = F_riccati(a, h[k+1], lambd, rho, nu)

    return t, h, f_vals