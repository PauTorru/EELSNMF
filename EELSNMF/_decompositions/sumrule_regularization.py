from ..imports import *
from ..utils import psi, convergent_psi, find_index


class LogSumRule_Regularization:
	
	def _LogSumRule_update_W(self,norm="mean"):
		HHt = self.H@self.H.T
		WHHt = self.W@HHt
		num = self.GtX@self.H.T 
		denum = self.GtG@WHHt+self.eps

		srgrad = self._LogSumRule_gradient()
		srgrad_pos = self.xp.maximum(srgrad,0)
		srgrad_neg = self.xp.maximum(-srgrad,0)
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)
		denum += self._norm*self.SR_lmbda*srgrad_pos
		num += self._norm*self.SR_lmbda*srgrad_neg
		self.W*=(num/denum)

# update H is the default one 
	
	def _calcB(self,edge):
		""" integral of original xsection over the fine structure energy range weighted by psi(E)"""
		if self.model._G_rescaled_to1:
			G0 = self._G0[:,self.model.xsection_idx[edge]]/self.model._scales[self.model.xsection_idx[edge]]
		else:
			G0 = self._G0[:,self.model.xsection_idx[edge]]
		ii,ff = find_index(self.energy_axis,self.fine_structure_ranges[edge])
		self.create_temp_array("_B_"+edge,(self.psi[ii:ff]*G0[ii:ff]).sum())
		return getattr(self,"_B_"+edge)


	
	def _LogSumRule_penalty(self):
		J = 0

		for i,edge in enumerate(self.edges):

			x = ((self._edge_psi[edge][:,None]*self.W[self._edge_indices[edge],:]).sum(0)+self.eps)/(self.B[edge]*self.W[self.model.xsection_idx[edge],:]+self.eps)
			J += (0.5*self.xp.log(x)**2).sum() #sums over components


		return J
	
	def _LogSumRule_gradient(self):
		self._dJcdW[:] = 0
		for edge in self.edges:
			v = self._edge_indices[edge]
			elnes_term = (self._edge_psi[edge][:,None]*self.W[v,:]).sum(0)
			i = self.model.xsection_idx[edge]
			xs_term = (self.B[edge]*self.W[i,:]+self.eps)

			x = elnes_term/xs_term
			log_ratio = self.xp.log(x)[None,:] 
			
			# elnes coeficients
			self._dJcdW [v,:] = log_ratio*self._edge_psi[edge][:,None]/elnes_term[None,:]

			# xs coeficients
			self._dJcdW[i,:] = - log_ratio/(self.W[i,:]+self.eps)

			# for edges below tolerance set gradient to 0:
			
			self._dJcdW[v,:]*=(self.xp.abs(log_ratio)>=self.SR_tolerance) # Tricky broadcasts.
			self._dJcdW[i,:]*=(self.xp.abs(log_ratio)>=self.SR_tolerance)[0,:]

		return  self._dJcdW

	

	def SumRule_decomposition(self,
		lmbda=0.1,
		norm="mean",
		SR_tolerance = 10.,
		convergent_beam_correction=False,
		convergent_factor_npoints=1000):
		"""Decomposition method to enforce sum_rules up to a tolerance


		Parameters
		----------

		lmbda: float
			regularization parameter: |X-GWH|+lmbda*(log(x))**2, x = sum( psi(e)*W_l,k)/B*W_xsection,k

		norm: {"mean","num","none"}
			normalization applied to the regularization:
				"num": lmbda = lmbda_0*G.T@X@H.T, element wise normalization updated each iteration
				"mean": lmbda = lmbda_0*(G.T@X@H.T).mean(), global normalization updated each iteration
				"none": no normalization
		SR_tolerance: float
			SumRule penalty is not applied for edges for which log(x)<log(SR_tolerance)

		convergent_beam_correction : bool
			Wether to apply convergent beam formulation for the energy weight function of the sum rule, formula (24) in https://doi.org/10.1016/j.ultramic.2024.114084

		convergent_factor_npoints: int
			See EELSNMF.utils.convergent_psi

			"""

		if SR_tolerance==0:
			self.SR_tolerance==0
		else:
			self.SR_tolerance = self.xp.log(SR_tolerance)
		self.get_model = self._default_get_model
		self._cbeam = convergent_beam_correction # for reporting purposes
		if convergent_beam_correction:
			self.psi = convergent_psi(self.energy_axis,
				self.alpha,
				self.beta,
				kV=self.E0,
				n_points=convergent_factor_npoints)
		else:
			self.psi = psi(self.energy_axis,
				self.beta,
				self.E0)

		self.SR_lmbda = lmbda
		self.B ={edge:self._calcB(edge) for edge in self.edges}
		self._edge_psi = {}
		for edge in self.edges:
			ii,ff = find_index(self.energy_axis,self.fine_structure_ranges[edge])
			self.create_temp_array("_"+edge+"_psi",self.psi[ii:ff])
			self._edge_psi[edge] = getattr(self,"_"+edge+"_psi")#self.psi[ii:ff]

		self._default_init_WH()
		self._build_S()
		self.enforce_dtype()
		#self.WS_reciprocal_sum=np.zeros_like(self.W)
		#self.W2 = self.W**2
		#self._m += ["WS_reciprocal_sum","W2"]
		
		

		self.create_temp_array("GtX",self.G.T@self.X)
		self.create_temp_array("GtG",self.G.T@self.G)
		self.create_temp_array("_dJcdW", np.zeros_like(self.W))
		

		if self.analysis_description["decomposition"]["use_cupy"]:
			self._np2cp()

		num = self.GtX@self.H.T 
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)

		error_0 = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.xp.linalg.norm(self._norm)*self.SR_lmbda*self._LogSumRule_penalty())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._LogSumRule_update_W(norm=norm)

				self.apply_fix_W()

				self._rescaleWH()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.xp.linalg.norm(self._norm)*self.SR_lmbda*self._LogSumRule_penalty())
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