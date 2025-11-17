import sympy
from scipy.differentiate import derivative

J, w = sympy.symbols('J,w')
J = w ** 3

dj_dw = sympy.diff(J, w)
print(dj_dw)

derivative_at_point = dj_dw.subs([(w, 2)])
print(derivative_at_point)
