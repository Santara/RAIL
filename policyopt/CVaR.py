import numpy as np 
import theano
from theano import tensor
from . import thutil
# from . import nn

class CVaR_params(object):
	"""docstring for CVaR_params"""
	def __init__(self, 
		Lambda_trainable, # Set this False if you want a min-max objective for CVaR rather than a constraint
		Lambda_val # This is used if Lambda_trainable=False
		): 
		self.Lambda_trainable = Lambda_trainable
		nu_initializer = np.random.random()
		Lambda_initializer = np.random.random()
		self.nu = theano.shared(name='nu', value=nu_initializer)
		if Lambda_trainable:
			self.Lambda = theano.shared(name='Lambda', value=Lambda_initializer)
		else:
			self.Lambda = theano.shared(name='Lambda', value=Lambda_val)
		# precomputed_updates = tensor.vector('CVaR_updates') #FIXME: 
		# params = [self.nu, self.Lambda]
		# self._learning_step = theano.function([precomputed_updates], [self.nu, self.Lambda], updates=zip(params,precomputed_updates))

	def fit(self, gradients, learning_rate=0.01):
		# updates = [self.nu - learning_rate*gradients[0], 
		# 			self.Lambda + learning_rate*gradients[1]]#gradient ascent
		# _,_=self._learning_step(updates)
		self.nu = self.nu - learning_rate*gradients[0] #We always 
		if self.Lambda_trainable:
			self.Lambda = self.Lambda + learning_rate*gradients[1]
		
