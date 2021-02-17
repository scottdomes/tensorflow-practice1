from sympy import symbols, diff

x, y, t = symbols('x y t', real=True)
f = t*x - y
result = diff(f,t)
print(result)
# t

f2 = t + x - y
result2 = diff(f2,t)
print(result2)
# x