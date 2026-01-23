from ..imports import *
from ..utils import find_index

class Frobenius_EdgeWisePenalty:
	
	def _FEWP_update_W(self):
		WHHt = self.W@self.H@self.H.T
		num = GtX@self.H.T 
		denum = GtG@WHHt+self.eps+self.FEWP_lmbda*self._FEWP_gradient()
		self.W*=(num/denum)

	def _FEWP_update_H(self,idxs,GtX,GtG):
		WH = self.W@self.H # full update
		num = self.W.T@GtX
		denum = self.W.T@GtG@WH+self.eps
		self.H*=num/denum

	def _FEWP(self):
		
		for edge in self.edges:
			pass


	def _FEWP_gradient(self):
		pass


	def _FEWP_decomposition(self,lmbda=0.1):
		
		self.FEWP_lmbda = lmbda
		self.get_model = self._default_get_model
		self._default_init_WH()
		
		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()
		
		if not hasattr(self,"GtX") or not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtG = self.G.T@self.G
			self.GtX = self.G.T@self.X
		
		if not "GtG" in self._m:
			self._m+=["GtG"]
		if not "GtX" in self._m:
			self._m+=["GtX"]




		self.enforce_dtype()

		error_0 = float(self.xp.abs(self.X-self.G@self.W@self.H).sum()+self.FEWP_lmbda*self._FEWP())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				for _ in range(iters_bg):
					self._alternate_update_W(idxs_bg,self.GtX_bg,self.GtG_bg)

					self.apply_fix_W()

					#self._alternate_update_H(idxs_bg,self.GtX_bg,self.GtG_bg)
				#self._alternate_update_H(idxs_elnes,self.GtX,self.GtG)
				for _ in range(iters_elnes):
					self._alternate_update_W2(idxs_elnes,self.GtX,self.GtG)

					self.apply_fix_W()

				self._alternate_update_H(idxs_elnes,self.GtX,self.GtG)
				
				if i%self.error_skip_step==0:
					error = float(self.xp.abs(self.X-self.G@self.W@self.H).sum())
					self.error_log.append(error)
					rel_change=float(self.xp.abs((error_0-error)/error_0))

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						self._cp2np()
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = self.xp.maximum(self.W, self.eps)
				self.H = self.xp.maximum(self.H, self.eps)

		for attr in ["GtX_bg","GtG_bg","GtX_elnes","GtG_elnes","GtG","GtX"]:
			if hasattr(self,attr):
				delattr(self,attr)
		if self.analysis_description["decomposition"]["use_cupy"]:
			self._cp2np()