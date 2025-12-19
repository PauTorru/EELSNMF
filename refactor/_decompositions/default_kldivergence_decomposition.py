from ..imports import *

class Default_KL:


	def _default_kl_update_W(self):
		self.GW = self.G@self.W
		np.divide(self.X,(self.GW@self.H+self.eps),out=self.X_over_GWH)
		temp = self.X_over_GWH@self.H.T
		num = self.G.T@temp
		denum = self.GTsum1[:,np.newaxis]@self.H.T.sum(0)[np.newaxis,:]+self.eps
		self.W*=num/denum
		

	def _default_kl_update_H(self):

		#self.X_over_GWH = self.X/(self.G@self.W@self.H+self.eps)
		num = self.GW.T@self.X_over_GWH
		denum = (self.GW.T).sum(1)[:,np.newaxis]+self.eps
		self.H*=num/denum


	def _default_kl_decomposition(self):
		
		self.get_model = self._default_get_model
		self._default_init_WH()
		self.KL_rescaling()
		self.GTsum1=self.G.T.sum(1)
		if not hasattr(self, "X_over_GWH"):
			self.X_over_GWH = np.empty_like(self.X)
		
		self.enforce_dtype()

		error_0 = self.KL_divergence_error()
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._default_kl_update_W()

				self.apply_fix_W()

				self._default_kl_update_H()
				
				if i%self.error_skip_step==0:
					error = self.KL_divergence_error()
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

				if self.rescale_WH:
					scale = self.W.sum(0,keepdims=True)
					scale = np.maximum(scale, self.eps)
					self.W /= scale
					self.H *= scale.T

				if self.KL_rescaling_per_iter:
					self.KL_rescaling()

	def KL_divergence_error(self):
		if hasattr(self,"GW"): #not the actual error (H is updated after computing self.GW) but faster to compute
			return (self.X * np.log((self.X + self.eps)/(self.GW@self.H + self.eps))
				- self.X + self.GW@self.H).sum()
		else:
			return (self.X * np.log((self.X + self.eps)/(self.G@self.W@self.H + self.eps))
				- self.X + self.G@self.W@self.H).sum()

	def KL_rescaling(self):
		scale = self.get_model().sum()/self.X.sum()
		self.H /= scale
		return