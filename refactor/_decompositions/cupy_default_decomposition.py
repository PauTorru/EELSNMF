import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf
from tqdm import tqdm
try:
	import cupy as cp
	CUPY_AVAILABLE = True
except:
	CUPY_AVAILABLE = False
	print("cupy not available")


class Cupy_Default:


	def _cupy_default_update_W(self):
		WHHt = cp.matmul(cp.matmul(self.W,self.H),cp.transpose(self.H))
		num = cp.matmul(self.GtX,cp.transpose(self.H))
		denum = cp.matmul(self.GtG,WHHt)+self.eps
		self.W = cp.multiply(self.W,cp.divide(num,denum))

	def _cupy_default_update_H(self):
		WH = cp.matmul(self.W,self.H)
		num = cp.matmul(cp.transpose(self.W),self.GtX)
		denum = cp.matmul(cp.matmul(cp.transpose(self.W),self.GtG),WH)+self.eps
		self.H = cp.multiply(self.H,cp.divide(num,denum))

	def _cupy_default_decomposition(self):
		
		self.get_model = self._default_get_model
		self._default_init_WH()# same as non cupy
		
		if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtX = self.G.T@self.X
			self.GtG = self.G.T@self.G

		if any(getattr(self,i).dtype!=self.dtype for i in self._m):
			self.change_dtype(self.dtype)

		self._np2cp()


		error_0 = float(cp.sum(cp.absolute(cp.subtract(self.X,
							cp.matmul(cp.matmul(self.G,self.W),self.H)))))
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._cupy_default_update_W()

				self.apply_fix_W()

				self._cupy_default_update_H()
				
				if i%self.error_skip_step==0:
					error = float(cp.sum(cp.absolute(cp.subtract(self.X,
							cp.matmul(cp.matmul(self.G,self.W),self.H)))))
					self.error_log.append(error)
					rel_change=abs((error_0-error)/error_0)

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						self._cp2np()
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = cp.maximum(self.W, self.eps)
				self.H = cp.maximum(self.H, self.eps)

		self._cp2np()

	def _np2cp(self):
		self.GtX = cp.array(self.GtX)
		self.GtG = cp.array(self.GtG)
		self.X = cp.array(self.X) 
		self.W = cp.array(self.W)
		self.G = cp.array(self.G)
		self.H = cp.array(self.H)

	def _cp2np(self):
		self.X = self.X.get()
		self.W = self.W.get()
		self.G = self.G.get()
		self.H = self.H.get()
		self.GtG  = self.GtG.get()
		self.GtX  = self.GtX.get()
