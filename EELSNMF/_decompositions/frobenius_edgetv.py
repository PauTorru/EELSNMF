from ..imports import *


class Frobenius_EdgeTV:
	
	def _EdgeTV_update_W(self,norm="mean"):
		HHt = self.H@self.H.T
		WHHt = self.W@HHt
		num = self.GtX@self.H.T 
		denum = self.GtG@WHHt+self.eps

		self._EdgeTV_gradient()
		
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)

		denum += self._norm*self.TV_lmbda*self._TVpos
		num += self._norm*self.TV_lmbda*self._TVneg
		self.W*=(num/denum)

		
# update H is the default one 

	
	def _EdgeTV(self):

		J = 0
		for edge,v in self._edge_indices.items():

			w = self.W[v,:]
			A = getattr(self,self._A_pos[edge])-getattr(self,self._A_neg[edge])
			J += 0.5*self.xp.sum(w*(A@w))

		return float(J)
	
	def _EdgeTV_gradient(self):

		self._TVpos[:]=0
		self._TVneg[:]=0

		for edge,v in self._edge_indices.items():
			w = self.W[v,:]
			self._TVpos[v,:]=getattr(self,self._A_pos[edge])@w
			self._TVneg[v,:]=getattr(self,self._A_neg[edge])@w

		return
		

		
		

	def _init_TV(self):
		self._A_pos = {}
		self._A_neg = {}
		for k,v in self._edge_indices.items():

			assert len(v)>3

			F = len(v)
			D2 = np.zeros((F-2,F))
			for i in range(F-2):
				D2[i,i] = 1.
				D2[i,i+1] = -2.
				D2[i,i+2] = 1.

			A = D2.T@D2
			self.create_temp_array("_A_pos_"+k,np.maximum(A,0.))
			self.create_temp_array("_A_neg_"+k,np.maximum(-A,0.))
			self._A_pos[k] = "_A_pos_"+k
			self._A_neg[k] = "_A_neg_"+k





	def EdgeTV_decomposition(self,lmbda=0.1,norm="mean"):

		self.TV_lmbda = lmbda
		self.get_model = self._default_get_model
		self._default_init_WH()
		self._build_S()
		self.enforce_dtype()
		

		self.create_temp_array("GtX",self.G.T@self.X)
		self.create_temp_array("GtG",self.G.T@self.G)
		self.create_temp_array("_TVpos", np.zeros_like(self.W))
		self.create_temp_array("_TVneg", np.zeros_like(self.W))
		
		self._init_TV()

		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

		num = self.GtX@self.H.T 
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)

		error_0 = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)**2+self.TV_lmbda*self._EdgeTV())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._EdgeTV_update_W(norm=norm)

				self.apply_fix_W()

				self._rescaleWH()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)**2+self.TV_lmbda*self._EdgeTV())
					self.error_log.append(error)
					rel_change=float(self.xp.abs((error_0-error)/error_0))

					if rel_change<=self.tol and i>2:
						print("Converged after {} iterations".format(i))
						if self.analysis_description["decomposition"]["use_cupy"]:
							self._cp2np()
						return
					
					pbar.set_postfix({"error":error,"relative change":rel_change})
					error_0 = error

					
				#shifts to prevent 0 locking
				self.W = self.xp.maximum(self.W, self.eps)
				self.H = self.xp.maximum(self.H, self.eps)

		self.delete_temp_arrays()


		if self.analysis_description["decomposition"]["use_cupy"]:
			self._cp2np()