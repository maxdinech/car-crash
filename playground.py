
# Descente de gradient 1D

def f(x):
	return x**2 + x

def df(x):
	return 2*x + 1

def descente_grad1(f, df, x, eta, epochs):
	for i in range(epochs):
		print(x)
		x = x - eta * df(x)**2
	print("Résultat :", x)

descente_grad1(f, df, 1, 0.1, 20)


# Descente de gradient 2D

def g(x, y):
	return x**2 + y**2

def dxg(x, y):
	return 2*x + y**2

def dyg(x, y):
	return x**2 + 2*y

def descente_grad2(g, dxg, dyg, x, y, eta, epochs):
	for i in range(epochs):
		print((x, y))
		(x, y) = (x - eta * dxg(x, y)**2, y - eta * dyg(x, y)**2)
	print("Résultat :", (x, y))

descente_grad2(f, dxf, dyf, 1, 1, 0.1, 10)


# Descente sur un seul neurone
import numpy as np

def f(w1, w2, w3, w4, b, x1, x2, x3, x4):
	return 1.0/(1.0 + np.exp(w1*x1+w2*x2+w3*x3+w4*x4-b))

def cost()