from ..imports import *

class Alternate_BG_ELNES:
	
	def _alternate_update_W(self,idxs,GtX,GtG):
		WHHt = self.W[idxs,:]@self.H@self.H.T
		num = GtX@self.H.T 
		denum = GtG@WHHt+self.eps
		self.W[idxs,:]*=num/denum

	def _alternate_update_H(self,idxs,GtX,GtG):
		WH = self.W[idxs,:]@self.H
		num = self.W[idxs,:].T@GtX
		denum = self.W[idxs,:].T@GtG@WH+self.eps
		self.H*=num/denum


	def _alternate_decomposition(self,iters_bg=1,iters_elnes=1):
		
		self.get_model = self._default_get_model
		self._default_init_WH()
		last_xsection = max(self.model.xsection_idx.values())
		idxs_bg = slice(None,last_xsection+1)
		idxs_elnes =slice(last_xsection+1,None)
		
		#if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
		self.GtX_bg = self.G[:,idxs_bg].T@self.X
		self.GtG_bg = self.G[:,idxs_bg].T@self.G[:,idxs_bg]
		self.GtX_elnes = self.G[:,idxs_elnes].T@self.X
		self.GtG_elnes = self.G[:,idxs_elnes].T@self.G[:,idxs_elnes]
		self._m+=["GtX_bg","GtX_elnes","GtG_bg","GtG_elnes"]

		self.enforce_dtype()

		error_0 = abs(self.X-self.G@self.W@self.H).sum()
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				for _ in range(iters_bg):
					self._alternate_update_W(idxs_bg,self.GtX_bg,self.GtG_bg)

					self.apply_fix_W()

					self._alternate_update_H(idxs_bg,self.GtX_bg,self.GtG_bg)
				
				for _ in range(iters_elnes):
					self._alternate_update_W(idxs_elnes,self.GtX_elnes,self.GtG_elnes)

					self.apply_fix_W()

					self._alternate_update_H(idxs_elnes,self.GtX_elnes,self.GtG_elnes)
				
				if i%self.error_skip_step==0:
					error = float(self.xp.abs(self.X-self.G@self.W@self.H).sum())
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

		for attr in ["GtX_bg","GtG_bg","GtX_elnes","GtG_elnes"]:
			if hasattr(self,attr):
				delattr(self,attr)
