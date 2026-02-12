from ..imports import *


class Cupy_Default_KL:

	def _cupy_default_kl_update_W(self):
		self.GW = self.G@self.W
		cp.divide(self.X,(self.GW@self.H+self.eps),out=self.X_over_GWH)
		temp = self.X_over_GWH@self.H.T
		num = self.G.T@temp
		denum = self.GTsum1[:,cp.newaxis]@self.H.T.sum(0)[cp.newaxis,:]+self.eps
		self.W*=num/denum
		

	def _cupy_default_kl_update_H(self):

		#self.X_over_GWH = self.X/(self.G@self.W@self.H+self.eps)
		num = self.GW.T@self.X_over_GWH
		denum = (self.GW.T).sum(1)[:,cp.newaxis]+self.eps
		self.H*=num/denum


	def _cupy_default_kl_decomposition(self,rescale_WH = False,KL_rescaling_per_iter = False):
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
		
		self.enforce_dtype()

		self._np2cp()
		self.GTsum1=self.G.T.sum(1)
		if not hasattr(self, "X_over_GWH"):
			self.X_over_GWH = cp.empty_like(self.X)

		
		error_0 = self._cupy_KL_divergence_error()
		self.error_log=[float(error_0)]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._cupy_default_kl_update_W()

				self.apply_fix_W()

				self._cupy_default_kl_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self._cupy_KL_divergence_error())
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

				if rescale_WH:
					scale = self.W.sum(0,keepdims=True)
					scale = cp.maximum(scale, self.eps)
					self.W /= scale
					self.H *= scale.T

				if KL_rescaling_per_iter:
					self.KL_rescaling()

		self._cp2np()

	def _cupy_KL_divergence_error(self):
		if hasattr(self,"GW"): #not the actual error (H is updated after computing self.GW) but faster to compute
			GWH=self.GW@self.H
			return (self.X * cp.log((self.X + self.eps)/(GWH + self.eps))
				- self.X + GWH).sum()
		else:
			GWH=self.G@self.W@self.H 
			return (self.X * cp.log((self.X + self.eps)/(GWH+ self.eps))
				- self.X + GWH).sum()
