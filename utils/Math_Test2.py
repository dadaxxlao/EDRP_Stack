from sympy import symbols, laplace_transform, Function, sin, cos, exp

# Define the symbols
t = symbols('t', real=True, positive=True)
s = symbols('s', real=True, positive=True)

# Define the function f(t)
f1 = 3*exp(-4*t)*sin(5*t)+17*exp(5*t)*cos(14*t)
f2 = 10*t**5+15
f3 = 27*exp(2*t)
f4 = 14*sin(7*t)+8*cos(13*t)
f5 = 14*exp(-5*t)*sin(7*t)+8*exp(4*t)*cos(13*t)
# Calculate the Laplace Transform of f(t)
F_s = laplace_transform(f5, t, s, noconds=True)
print(F_s)