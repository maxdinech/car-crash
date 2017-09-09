import random
import numpy as np


def sigmoide(z):
	return 1.0/(1.0+np.exp(-z))

def d_sigmoide(z):
	return sigmoid(z)*(1-sigmoid(z))



class Réseau(object):

	def __init__(self, tailles_couches):
		self.nb_couches = len(tailles_couches)
		self.tailles_couches = tailles_couches
		self.biases = [np.random.randn(y, 1) for y in tailles_couches[1:]]
		self.weights = [np.random.randn(y, x)
					  for x, y in zip(tailles_couches[:-1], tailles_couches[1:])]

	def propagation(self, a):
		"""Propage le vecteur a dans le réseau"""
		for b, w in zip(self.biases, self.weights):
			a = self.activation(np.dot(w, a) + b)
		return a

	def SGD(self, training_data, epochs, taille_mini_batch, η, test_data = None):
		training_data = list(training_data)
		n = len(training_data)

		if test_data:
			test_data = list(test_data)
			n_test = len(test_data)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+taille_mini_batch]
				for k in range(0, n, taille_mini_batch)]
			for mini_batch in mini_batches:
				self.descente_locale(mini_batch, η)
			if test_data:
				print("Epoch {} : {}/{}".format(j, self.succès(test_data), n_test))
			else:
				print("Epoch {} terminé".format(j))

	def descente_locale(self, mini_batch, η):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for b in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.retroprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(η/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(η/len(mini_batch))*nb
					   for b, nb in zip(self.biases, nabla_b)]

	def retroprop(self, x, y):
		"""Descente de gradient sur une seule valeur"""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
			d_sigmoide(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in range(2, self.nb_couches):
			z = zs[-l]
			sp = d_sigmoide(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def succès(self, test_data):
		"""Calcule le taux de succès du réseau sur ``data``."""
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)
