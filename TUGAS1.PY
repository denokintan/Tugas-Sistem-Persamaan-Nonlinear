import math
import numpy as np
import pandas as pd

def F(vec):
    x, y = vec
    return np.array([x**2 + x*y - 10, y + 3*x*y**2 - 57], dtype=float)

def jacobian(vec):
    x, y = vec
    return np.array([[2*x + y, x],
                     [3*y**2, 1 + 6*x*y]], dtype=float)

def g1A(x, y): 
    return (-y + math.sqrt(y**2 + 40)) / 2

def g1B(x, y):
    return (-y - math.sqrt(y**2 + 40)) / 2

def g2A(x, y): 
    if x == 0: return float('nan')
    disc = 1 + 684*x
    if disc < 0: return float('nan')
    return (-1 + math.sqrt(disc)) / (6*x)

def g2B(x, y):
    if x == 0: return float('nan')
    disc = 1 + 684*x
    if disc < 0: return float('nan')
    return (-1 - math.sqrt(disc)) / (6*x)

def fixed_point_iter(gx, gy, x0, y0, tol=1e-6, maxit=200, method='Jacobi'):
    x, y = x0, y0
    for k in range(1, maxit+1):
        if method == 'Jacobi':
            x_new = gx(x, y)
            y_new = gy(x, y)
        else:  # Seidel
            x_new = gx(x, y)
            y_new = gy(x_new, y)
        if any(math.isnan(v) for v in [x_new, y_new]):
            return {'converged': False, 'iter': k, 'x': x_new, 'y': y_new}
        if max(abs(x_new - x), abs(y_new - y)) < tol:
            return {'converged': True, 'iter': k, 'x': x_new, 'y': y_new}
        x, y = x_new, y_new
    return {'converged': False, 'iter': maxit, 'x': x, 'y': y}

def newton(vec0, tol=1e-6, maxit=200):
    x = np.array(vec0, dtype=float)
    for k in range(1, maxit+1):
        J = jacobian(x)
        Fv = F(x)
        delta = np.linalg.solve(J, -Fv)
        x_new = x + delta
        if np.max(np.abs(delta)) < tol:
            return {'converged': True, 'iter': k, 'x': x_new}
        x = x_new
    return {'converged': False, 'iter': maxit, 'x': x}

def broyden(vec0, tol=1e-6, maxit=200):
    x = np.array(vec0, dtype=float)
    B = jacobian(x) 
    Fv = F(x)
    for k in range(1, maxit+1):
        p = np.linalg.solve(B, -Fv)
        x_new = x + p
        F_new = F(x_new)
        s = p
        y = F_new - Fv
        denom = np.dot(s, s)
        if denom == 0: break
        B = B + np.outer((y - B.dot(s)), s) / denom
        if np.max(np.abs(p)) < tol:
            return {'converged': True, 'iter': k, 'x': x_new}
        x, Fv = x_new, F_new
    return {'converged': False, 'iter': maxit, 'x': x}



x0, y0 = 1.5, 3.5
tol = 1e-6
results = []

combos = [
    ('g1A', 'g2A', g1A, g2A),
    ('g1A', 'g2B', g1A, g2B),
    ('g1B', 'g2A', g1B, g2A),
    ('g1B', 'g2B', g1B, g2B)
]

for name1, name2, gx, gy in combos:
    for method in ['Jacobi', 'Seidel']:
        out = fixed_point_iter(gx, gy, x0, y0, tol=tol, maxit=200, method=method)
        results.append({
            'Method': f'IT-{method} ({name1},{name2})',
            'Converged': out['converged'],
            'Iterations': out['iter'],
            'x': out['x'],
            'y': out['y']
        })


nr = newton((x0, y0), tol=tol)
results.append({
    'Method': 'Newton-Raphson',
    'Converged': nr['converged'],
    'Iterations': nr['iter'],
    'x': nr['x'][0],
    'y': nr['x'][1]
})


br = broyden((x0, y0), tol=tol)
results.append({
    'Method': 'Secant',
    'Converged': br['converged'],
    'Iterations': br['iter'],
    'x': br['x'][0],
    'y': br['x'][1]
})

df = pd.DataFrame(results)
print(df.to_string(index=False))
