#!/usr/bin/python2.6
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: nmf.py 20 2010-08-02 17:35:19Z cthurau $
#$Author: cthurau $
"""  	
PyMF Non-negative Matrix Factorization.

	NMF: Class for Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative 
Matrix Factorization, Nature 401(6755), 788-799.
"""

__version__ = "$Revision: 8 $"


import time
import sys
import random
import numpy as np
from progressbar import ProgressBar

__all__ = ["NMF"]

class NMF:
	"""  	
	NMF(data, num_bases=4, niter=100, show_progress=False, compW=True)
	
	
	Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
	s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
	data. Uses the classicial multiplicative update rule.
	
	Parameters
	----------
	data : array_like
		the input data
	num_bases: int, optional 
		Number of bases to compute (column rank of W and row rank of H). 
		4 (default)
	niter: int, optional
		Number of iterations of the alternating optimization.
		100 (default)
	show_progress: bool, optional
		Print some extra information
		False (default)
	compW: bool, optional
		Compute W (True) or only H (False). Useful for using precomputed
		basis vectors.
	
	Attributes
	----------
		W : "data_dimension x num_bases" matrix of basis vectors
		H : "num bases x num_samples" matrix of coefficients	
	
	Example
	-------
	Applying NMF to some rather stupid data set:
	
	>>> import numpy as np
	>>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
	>>> nmf_mdl = NMF(data, num_bases=2, niter=10)
	>>> nmf_mdl.initialization()
	>>> nmf_mdl.factorize()
	
	The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
	To compute coefficients for an existing set of basis vectors simply	copy W 
	to nmf_mdl.W, and set compW to False:
	
	>>> data = np.array([[1.5], [1.2]])
	>>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
	>>> nmf_mdl = NMF(data, num_bases=2, niter=1, compW=False)
	>>> nmf_mdl.initialization()
	>>> nmf_mdl.W = W
	>>> nmf_mdl.factorize()
	
	The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
	"""
	
	_VINFO = 'pymf-nmf v0.1'
	
	EPS = 10**-8
	
	def __init__(self, data, num_bases=4, niter=100, show_progress=False, compH=True, compW=True):
		
		self.data = data		
		self._num_bases = num_bases
		self._niter = niter
		self._show_progress = show_progress
		self.ferr = np.zeros(self._niter)
		
		# initialize H and W to random values
		(self._data_dimension, self._num_samples) = self.data.shape			
		
		# control if W should be updated -> usefull for assigning precomputed basis vectors
		self._compW = compW
		self._compH = compH

	def _prog_bar(self, total):
		if total > 1:
			self._prog = ProgressBar(0, total, 77, mode='fixed', char='#')
		else:
			self._prog = ProgressBar(0, 1, 77, mode='fixed', char='#')				
		
		
	def _update_prog_bar(self):
		if self._show_progress:
			self._prog.increment_amount()
			sys.stdout.write("\r")				
			sys.stdout.write(str(self._prog))
			sys.stdout.write("\n")
			sys.stdout.flush()
				
				
	def _print_cur_status(self, message):		
		if self._show_progress:
			sys.stdout.write('[' + time.ctime() + '/' + self._VINFO + '] ' + message + '\n')					
	
	
	def initialization(self):
		""" Initialize W and H to random values in [0,1].
		"""
		# init
		self.H = np.random.random((self._num_bases, self._num_samples))
		self.W = np.random.random((self._data_dimension, self._num_bases))
		# set W to some random data samples
		sel = random.sample(xrange(self._num_samples), self._num_bases)
		
		# sort indices, otherwise h5py won't work
		self.W = self.data[:, np.sort(sel)]

	def frobenius_norm(self):
		""" Frobenius norm (||data - WH||) for a data matrix and a low rank
		approximation given by WH
		
		Returns:
			frobenius norm: F = ||data - WH||
		"""		
	
		err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))
		return err


	def updateH(self):
			# pre init H1, and H2 (necessary for storing matrices on disk)									
			H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
			self.H *= np.dot(self.W.T, self.data[:,:])
			self.H /= H2				
	
	def updateW(self):
			# pre init W1, and W2 (necessary for storing matrices on disk)									
			W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
			self.W *= np.dot(self.data[:,:], self.H.T)
			self.W /= W2		

	def converged(self, i):
		derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
		if derr < self.EPS:
			return True
		else:
			return False
		
	def factorize(self):
		""" Perform factorization s.t. data = WH using the standard multiplicative 
		update rule for Non-negative matrix factorization.
		"""	
							
		# iterate over W and H
		for i in xrange(self._niter):						
			# update W
			if self._compW:
				self.updateW()
									
			# update H
			self.updateH()
								
			self.ferr[i] = self.frobenius_norm()		
											
			self._print_cur_status('iteration ' + str(i+1) + '/' + str(self._niter) + ' Fro:' + str(self.ferr[i]))
			
					# check if the err is not changing anymore
			if i > 1:
				if self.converged(i):		
					# adjust the error measure
					self.ferr = self.ferr[:i]			
					break
				
if __name__ == "__main__":
	import doctest  
	doctest.testmod()	