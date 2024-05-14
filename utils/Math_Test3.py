from sympy import symbols, apart, inverse_laplace_transform, exp

# Define the symbols
s, t = symbols('s t')
Y_s1 = 24/((s+11)*(s-13))
Y_s2 = -324/(3*s**3+27*s)
Y_s3 = 4*s/((s+4)*(s+6)**2)
Y_s4 = (5*s+17)/((s-2)*(s+7))
Y = 288/(s**3-6*s**2+32)
Y2 = (-6*exp(-s)+60+12*s)/(s**2+4*s+13)
Y3 = (16-20*exp(-3*s))/(4*s**2+16*s+20)

# Perform partial fraction decomposition
#0Y_s_apart = apart(Y3)

# Compute the Inverse Laplace Transform
y_t = inverse_laplace_transform(Y3, s, t)
print(y_t)
