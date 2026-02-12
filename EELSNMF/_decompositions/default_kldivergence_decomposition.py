from ..imports import *

class Default_KL:


	def _default_kl_update_W(self):
		self.create_temp_array("GW",self.G@self.W)
		self.xp.divide(self.X,(self.GW@self.H+self.eps),out=self.X_over_GWH)
		temp = self.X_over_GWH@self.H.T
		num = self.G.T@temp
		denum = self.GTsum1[:,None]@self.H.T.sum(0)[None,:]+self.eps
		self.W*=num/denum
		

	def _default_kl_update_H(self):

		#self.X_over_GWH = self.X/(self.G@self.W@self.H+self.eps)
		num = self.GW.T@self.X_over_GWH
		denum = (self.GW.T).sum(1)[:,None]+self.eps
		self.H*=num/denum


	def _default_kl_decomposition(self,rescale_WH = False, KL_rescaling_per_iter = False):
		"""
			rescale_WH : bool
				Only used for metric="KLdivergence"
				Rescales columns of W to one.
				 (Default value = False)
			KL_rescaling_per_iter :
				Only used for metric="KLdivergence". Rescales the model to accurately capture absolute intensity at each iteration.
				 (Default value = False)
				 """
		
		self.get_model = self._default_get_model
		self._default_init_WH()
		self.KL_rescaling()
		self.create_temp_array("GTsum1",self.G.T.sum(1))
		self.create_temp_array("X_over_GWH",self.xp.empty_like(self.X))

		self.enforce_dtype()
		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

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
					rel_change=self.xp.abs((error_0-error)/error_0)

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = self.xp.maximum(self.W, self.eps)
				self.H = self.xp.maximum(self.H, self.eps)

				if rescale_WH:
					scale = self.W.sum(0,keepdims=True)
					scale = self.xp.maximum(scale, self.eps)
					self.W /= scale
					self.H *= scale.T

				if KL_rescaling_per_iter:
					self.KL_rescaling()

		if self.analysis_description["decomposition"]["use_cupy"]:
			self._cp2np()

		self.delete_temp_arrays()

	def KL_divergence_error(self):
		if hasattr(self,"GW"): #not the actual error (H is updated after computing self.GW) but faster to compute
			return (self.X * self.xp.log((self.X + self.eps)/(self.GW@self.H + self.eps))
				- self.X + self.GW@self.H).sum()
		else:
			return (self.X * self.xp.log((self.X + self.eps)/(self.G@self.W@self.H + self.eps))
				- self.X + self.G@self.W@self.H).sum()

	def KL_rescaling(self):
		scale = self.get_model().sum()/self.X.sum()
		self.H /= scale
		return