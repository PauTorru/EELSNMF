
from ._decompositions.default_decomposition import Default
from ._decompositions.cupy_default_decomposition import Cupy_Default
from ._decompositions.default_kldivergence_decomposition import Default_KL
from ._decompositions.cupy_default_kldivergence_decomposition import Cupy_Default_KL
from ._decompositions.cupy_utils import Cupy_Utils
from ._decompositions.alternate_bg_elnes import Alternate_BG_ELNES
from .imports import *


class Decomposition(Default, Cupy_Default, Default_KL, Cupy_Default_KL,Cupy_Utils):
	_DECOMPOSITION_CHOICES = {#(model_type,use_cupy)
		("deltas",False,"Frobenius"):"_default_decomposition",
		("deltas",True,"Frobenius"):"_cupy_default_decomposition",
		("convolved_single",False,"Frobenius"):"_default_decomposition",
		("convolved_single",True,"Frobenius"):"_cupy_default_decomposition",
		("deltas",False,"KLdivergence"):"_default_kl_decomposition",
		("deltas",True,"KLdivergence"):"_cupy_default_kl_decomposition",
		("convolved_single",False,"KLdivergence"):"_default_kl_decomposition",
		("convolved_single",True,"KLdivergence"):"_cupy_default_kl_decomposition",
		}

	def decomposition(self,n_components,
								max_iters = 100,
								tol = 1e-6,
								use_cupy = False,
								init_nmf = None,
								random_state_nmf = None,
								W_init = None,
								W_fixed_bool = None,
								W_fixed_values = None,
								H_init = None,
								error_skip_step=10,
								eps=1e-10,
								rescale_WH=False,
								KL_rescaling_per_iter = False,
								metric = "Frobenius",
								decomposition_method = None,
								**kwargs):
		

		self.n_components = n_components
		self.max_iters = max_iters
		self.tol =tol
		self.error_skip_step = error_skip_step
		self.init_nmf=init_nmf
		self.eps=eps
		self.rescale_WH = rescale_WH
		self.metric=metric
		self.KL_rescaling_per_iter = KL_rescaling_per_iter


		self.analysis_description["decomposition"]={}

		if use_cupy and not CUPY_AVAILABLE:
			print("cupy not available; falling back to cpu")
		self.analysis_description["decomposition"]["use_cupy"] = use_cupy and CUPY_AVAILABLE
		if use_cupy and CUPY_AVAILABLE:
			self.xp = cp
		else:
			self.xp = np

		self.analysis_description["decomposition"]["Fix_W"] = not W_fixed_bool is None
		self.analysis_description["decomposition"]["metric"] = self.metric

		self.random_state_nmf=random_state_nmf
		self.W_init=W_init
		self.W_fixed_bool=W_fixed_bool
		self.W_fixed_values=W_fixed_values
		self.H_init = H_init
		if metric == "KLdivergence":
			self._m += ["GW","X_over_GWH","GTsum1"]
		elif metric == "Frobenius":
			self._m += ["GtG","GtX"]
		self._m +=["W_init","W_fixed_values","H_init"]
		self._m = list(set(self._m))

		if decomposition_method is None:
			decomposition_method = self._DECOMPOSITION_CHOICES[(self.analysis_description["model_type"],self.analysis_description["decomposition"]["use_cupy"],self.analysis_description["decomposition"]["metric"])]
		
		self.analysis_description["decomposition"]["method"]=decomposition_method
		method = getattr(self,decomposition_method)
		method(**kwargs)
		
		#clear memory
		for attr in ["GtX","GtG","GW","X_over_GWH","GTsum1"]:
			if hasattr(self,attr):
				delattr(self,attr)
			if attr in self._m:
				self._m.remove(attr)
		gc.collect()

	def apply_fix_W(self):
		if not self.W_fixed_bool is None:
			self.W[self.W_fixed_bool]=self.W_fixed_values[self.W_fixed_bool]

	def _default_get_model(self):
		return self.G@self.W@self.H

	def enforce_dtype(self):
		for attr in self._m:
			if hasattr(self,attr):
				value = getattr(self,attr,None)
				if value is not None and hasattr(value,"dtype"):
					if value.dtype!= self.dtype:
						setattr(self,attr,value.astype(self.dtype,copy=False))

	def _default_init_WH(self):

		assert not self.G is None
		if not self.W_init is None:
			self.W = self.W_init.copy()
			if self.H_init is None:
				self.H = np.abs(np.linalg.lstsq(self.G@self.W, self.X,rcond=None)[0])
			else:
				self.H = self.H_init.copy()
		elif self.H_init is None:
			GW,self.H = initialize_nmf(self.X,self.n_components,init=self.init_nmf,random_state=self.random_state_nmf)
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])

		else: # no W_init, yes H_init
			self.H = self.H_init.copy()
			GW = np.linalg.lstsq(self.H.T,self.X.T,rcond=None)[0].T
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])


		self.H = np.nan_to_num(self.H, nan=self.eps, posinf=self.eps, neginf=self.eps)
		self.W = np.nan_to_num(self.W, nan=self.eps, posinf=self.eps, neginf=self.eps)

		








