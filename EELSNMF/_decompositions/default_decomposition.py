from ..imports import *

class Default:
	
	def _default_update_W(self):
		WHHt = self.W@self.H@self.H.T
		num = self.GtX@self.H.T 
		denum = self. GtG@WHHt+self.eps
		self.W*=num/denum

	def _default_update_H(self):
		WH = self.W@self.H
		num = self.W.T@self.GtX
		denum = self.W.T@self.GtG@WH+self.eps
		self.H*=num/denum


	def _default_decomposition(self):
		
		self.get_model = self._default_get_model
		self._default_init_WH()
		
		if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtX = self.G.T@self.X
			self.GtG = self.G.T@self.G

		if not "GtG" in self._m:
			self._m+=["GtG"]
		if not "GtX" in self._m:
			self._m+=["GtX"]

		self.enforce_dtype()

		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

		error_0 = self.xp.abs(self.X-self.G@self.W@self.H).sum()
		self.error_log=[float(error_0)]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._default_update_W()

				self.apply_fix_W()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = self.xp.abs(self.X-self.G@self.W@self.H).sum()
					self.error_log.append(float(error))
					rel_change=self.xp.abs((error_0-error)/error_0)

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = self.xp.maximum(self.W, self.eps)
				self.H = self.xp.maximum(self.H, self.eps)


		for attr in ["GtG","GtX"]:
				if hasattr(self,attr):
					delattr(self,attr)
			if self.analysis_description["decomposition"]["use_cupy"]:
				self._cp2np()


	