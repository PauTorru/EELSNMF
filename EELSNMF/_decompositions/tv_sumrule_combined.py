from ..imports import *
from ..utils import psi, convergent_psi, find_index

class TV_SumRule:

	def _TVSR_update_W(self,norm="mean"):
		HHt = self.H@self.H.T
		WHHt = self.W@HHt
		num = self.GtX@self.H.T 
		denum = self.GtG@WHHt+self.eps

		srgrad = self._LogSumRule_gradient()
		srgrad_pos = self.xp.maximum(srgrad,0)
		srgrad_neg = self.xp.maximum(-srgrad,0)

		tvgrad = self._EdgeTV_gradient()
		TVgrad_pos = self._TV_majorizer/(self.W+self.eps)# for quadratic -> self.W*self._TV_majorizer #-> ensures convergence instead of self.xp.maximum(TVgrad,0)
		TVgrad_neg = self.xp.maximum(-TVgrad,0)
		
		if norm == "mean":
			self._norm = self.xp.mean(num)
		elif norm == "num":
			self.create_temp_array("_norm", num)
		elif norm == "none":
			self._norm = self.xp.array(1.)
		
		denum += self._norm*(self.SR_lmbda*srgrad_pos+self.TV_lmbda*TVgrad_pos)
		num += self._norm*(self.SR_lmbda*srgrad_neg+self.TV_lmbda*TVgrad_neg)
		self.W*=(num/denum)

	def combinedTVSR_decomposition(self,
		TV_lmbda=0.1,
		SR_lmbda=0.1,
		norm="mean",
		inertia_dJdW=1,
		clip=1.,
		SR_tolerance = 10.,
		convergent_beam_correction = False,
		convergent_factor_npoints = 1000 ):
		"""Decomposition method enforcing sum_rules and total variation minimization.


		Parameters
		----------
		
		TV_lmbda: float
			regularization parameter: TV_lmbda*sum_edges(TV(edge))
		
		SR_lmbda: float
			regularization parameter: SR_lmbda*(log(x))**2, x = sum( psi(e)*W_l,k)/B*W_xsection,k
		
		norm: {"mean","num","none"}
			normalization applied to the regularization:
				"num": lmbda = lmbda_0*G.T@X@H.T, element wise normalization updated each iteration
				"mean": lmbda = lmbda_0*(G.T@X@H.T).mean(), global normalization updated each iteration
				"none": no normalization
		
		inertia_dJdW: float
			inertia parameter for the TV gradient (gradient_i = (gradient_i+inertia*gradient_i-1))/(1+inertia)

		clip: float
			abs(TV gradient) is clipped to this value
		
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
		self.inertia_dJdW = inertia_dJdW
		self.TV_lmbda = TV_lmbda
		self._dJdWclip = clip

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

		self.SR_lmbda = SR_lmbda
		self.B ={}
		for edge in self.edges:
			self.B[edge]=self._calcB(edge) # generating dict with comprehension breaks cp2np
		self._edge_psi = {}
		for edge in self.edges:
			ii,ff = find_index(self.energy_axis,self.fine_structure_ranges[edge])
			self.create_temp_array("_"+edge+"_psi",self.psi[ii:ff])
			self._edge_psi[edge] = "_"+edge+"_psi"


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
		self.create_temp_array("_dJcdW", np.zeros_like(self.W))#sumrule gradient
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

		error_0 = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.TV_lmbda*self._EdgeTV()+self.SR_lmbda*self._LogSumRule_penalty())
		self.error_log=[error_0]

		with tqdm(range(self.max_iters),mininterval=5) as pbar:
			for i in pbar:

				self._TVSR_update_W(norm=norm)

				self.apply_fix_W()

				self._rescaleWH()

				self._default_update_H()
				
				if i%self.error_skip_step==0:
					error = float(self.xp.linalg.norm(self.X-self.G@self.W@self.H)+self.xp.linalg.norm(self._norm)*(self.TV_lmbda*self._EdgeTV()+self.SR_lmbda*self._LogSumRule_penalty()))
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


	
