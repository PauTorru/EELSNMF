from ..imports import *


class Frobenius_EdgeTV:
	
	def _EdgeTV_update_W(self,norm="mean"):
		HHt = self.H@self.H.T
		WHHt = self.W@HHt
		num = self.GtX@self.H.T 
		denum = self.GtG@WHHt+self.eps

		TVgrad = self._EdgeTV_gradient()
		TVgrad_pos = self.W*self._TV_majorizer #-> ensures convergence instead of self.xp.maximum(TVgrad,0) 
		TVgrad_neg = self.xp.maximum(-TVgrad,0)
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)
		denum += self._norm*self.TV_lmbda*TVgrad_pos
		num += self._norm*self.TV_lmbda*TVgrad_neg
		self.W*=(num/denum)

		
# update H is the default one 

	
	def _EdgeTV(self):

		J = 0
		for k,v in self._edge_indices.items():

			J += self.xp.abs(self.W[v[1:],:]-self.W[v[:-1],:]).sum() # sums contributions for all components together. Strictily first should sum over axis=0 (J of edge of each component) and then over axis=1.

		return J
	
	def _EdgeTV_gradient(self):
		self.old_dJdW[:] = self._dJdW[:]

		for k,v in self._edge_indices.items():
			diffs = self.xp.diff(self.W[v,:],axis=0)
			#smooth_signs = diffs/self.xp.sqrt(diffs**2+eps)  for L1 TV 
			#self._dJdW[v[1:-1],:] = smooth_signs[:-1,:]-smooth_signs[1:,:]
			#self._dJdW[v[0],:] = -smooth_signs[0,:]
			#self._dJdW[v[-1],:] = smooth_signs[-1,:]
			self._dJdW[v[1:-1],:] = diffs[:-1,:]-diffs[1:,:] # for L2 TV
			self._dJdW[v[0],:] = -diffs[0,:]
			self._dJdW[v[-1],:] = diffs[-1,:]

		self._dJdW = self.xp.clip((self._dJdW + self.inertia_dJdW*self.old_dJdW)/(1+self.inertia_dJdW),-self._dJdWclip,self._dJdWclip)
		return self._dJdW

	def _init_TVmajorizer(self):
		self.create_temp_array("_TV_majorizer",np.zeros_like(self.W))
		for v in self._edge_indices.values():

			assert len(v)>2

			# endpoints
			self._TV_majorizer[v[0],:] = 2
			self._TV_majorizer[v[-1],:] = 2
			#interior
			self._TV_majorizer[v[1:-1],:] = 4 

	def EdgeTV_decomposition(self,lmbda=0.1,norm="mean",inertia_dJdW=1,clip=1.):

		self.inertia_dJdW = inertia_dJdW
		self.TV_lmbda = lmbda
		self._dJdWclip = clip#1#2*self.X.std()**2
		self.get_model = self._default_get_model
		self._default_init_WH()
		self._build_S()
		self.enforce_dtype()
		#self.WS_reciprocal_sum=np.zeros_like(self.W)
		#self.W2 = self.W**2
		#self._m += ["WS_reciprocal_sum","W2"]
		
		

		self.create_temp_array("GtX",self.G.T@self.X)
		self.create_temp_array("GtG",self.G.T@self.G)
		self.create_temp_array("_dJdW", np.zeros_like(self.W))
		self.create_temp_array("old_dJdW",self._dJdW.copy())
		self._init_TVmajorizer()
		
		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

		num = self.GtX@self.H.T 
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)

		error_0 = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.TV_lmbda*self._EdgeTV())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._EdgeTV_update_W(norm=norm)

				self.apply_fix_W()

				self._rescaleWH()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.xp.linalg.norm(self._norm)*self.TV_lmbda*self._EdgeTV())
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