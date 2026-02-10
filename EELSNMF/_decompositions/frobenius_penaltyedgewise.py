from ..imports import *
from ..utils import find_index

class Frobenius_PenaltyEdgeWise:
	
	def _FPEW_update_W(self):
		HHt = self.H@self.H.T
		WHHt = self.W@HHt
		num = self.GtX@self.H.T 
		denum = self.GtG@WHHt+self.eps
		#self._normalization = num.mean()
		denum += #self._normalization*self.FPEW_lmbda*self._FPEW_gradient()
		self.W*=(num/denum)

	def _FPEW_update_H(self):
		WH = self.W@self.H # full update
		num = self.W.T@self.GtX
		denum = self.W.T@self.GtG@WH+self.eps
		self.H*=num/denum

"""	def _FPEW(self):

		if not hasattr(self,"_edge_indices"):
			self._build_S()

		out = 0
		for edge,v in self._edge_indices.items():
			out += self.xp.linalg.norm(self.W[v,:],axis=0).sum()

		return out

	def _FPEW_gradient(self):

		self.WS_reciprocal_sum[:]=0

		for edge,v in self._edge_indices.items():
			self.WS_reciprocal_sum[v,:] +=  1/self.xp.sqrt((self.W2[v,:]).sum(0)+self.eps) # += 1/(np.linalg.norm(self.W[v,:].axis=0)[None,:]+self.eps) not sure what expression is better

		return self.W*self.WS_reciprocal_sum
"""

	

	def _FPEW_decomposition(self,lmbda=0.1):
		
		self.FPEW_lmbda = lmbda
		self.get_model = self._default_get_model
		self._default_init_WH()
		self._build_S()
		self.enforce_dtype()
		#self.WS_reciprocal_sum=np.zeros_like(self.W)
		#self.W2 = self.W**2
		#self._m += ["WS_reciprocal_sum","W2"]
		
		if not hasattr(self,"GtX") or not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtG = self.G.T@self.G
			self.GtX = self.G.T@self.X
		
		if not "GtG" in self._m:
			self._m+=["GtG"]
		if not "GtX" in self._m:
			self._m+=["GtX"]

		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

		num = self.GtX@self.H.T 
		#self._normalization = num.mean()
		error_0 = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self._normalization*self.FPEW_lmbda*self._FPEW())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._FPEW_update_W()

				self.apply_fix_W()

				self._FPEW_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self._normalization*self.FPEW_lmbda*self._FPEW())
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
				self.W2=self.W**2

		for attr in ["GtG","GtX","WS_reciprocal_sum","W2"]:
			if hasattr(self,attr):
				delattr(self,attr)
		if self.analysis_description["decomposition"]["use_cupy"]:
			self._cp2np()