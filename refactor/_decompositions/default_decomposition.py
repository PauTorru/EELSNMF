from ..imports import *

class Default:
	def _default_init_WH(self):

		assert not self.G is None
		if not self.W_init is None:
			self.W = self.W_init.copy()
			if self.H_init is None:
				self.H = np.abs(np.linalg.lstsq(self.G@self.W, self.X,rcond=None)[0])
			else:
				self.H = self.H_init.copy()
		elif self.H_init is None:
			GW,self.H = initialize_nmf(self.X,self.n_components,init=self.init_nmf,random_state=self.random_state_nmf)
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])

		else: # no W_init, yes H_init
			self.H = self.H_init.copy()
			GW = np.linalg.lstsq(self.H.T,self.X.T,rcond=None)[0].T
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])


		self.H = np.nan_to_num(self.H, nan=self.eps, posinf=self.eps, neginf=self.eps)
		self.W = np.nan_to_num(self.W, nan=self.eps, posinf=self.eps, neginf=self.eps)

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

		if any(getattr(self,i).dtype!=self.dtype for i in self._m):
			self.change_dtype(self.dtype)

		error_0 = abs(self.X-self.G@self.W@self.H).sum()
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._default_update_W()

				self.apply_fix_W()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = abs(self.X-self.G@self.W@self.H).sum()
					self.error_log.append(error)
					rel_change=abs((error_0-error)/error_0)

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = np.maximum(self.W, self.eps)
				self.H = np.maximum(self.H, self.eps)


	def _default_get_model(self):
		return self.G@self.W@self.H