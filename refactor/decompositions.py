import numpy as np
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf
from tqdm import tqdm
from ._decompositions.default_decomposition import Default
from ._decompositions.cupy_default_decomposition import Cupy_Default
try:
	import cupy as cp
	CUPY_AVAILABLE = True
except:
	CUPY_AVAILABLE = False
	print("cupy not available")


class Decomposition(Default, Cupy_Default):
	_DECOMPOSITION_CHOICES = {#(model_type,use_cupy)
		("deltas",False):"_default_decomposition",
		("deltas",True):"_cupy_default_decomposition",
		("convolved_single",False):"_default_decomposition",
		("convolved_single",True):"_cupy_default_decomposition",
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
								eps=1e-10):
		

		self.n_components = n_components
		self.max_iters = max_iters
		self.tol =tol
		self.error_skip_step = error_skip_step
		self.init_nmf=init_nmf
		self.eps=eps


		self.analysis_description["decomposition"]={}

		if use_cupy and not CUPY_AVAILABLE:
			print("cupy not available; falling back to cpu")
		self.analysis_description["decomposition"]["use_cupy"] = use_cupy and CUPY_AVAILABLE

		self.analysis_description["decomposition"]["Fix_W"] = not W_fixed_bool is None


		self.random_state_nmf=random_state_nmf
		self.W_init=W_init
		self.W_fixed_bool=W_fixed_bool
		self.W_fixed_values=W_fixed_values
		self.H_init = H_init

		decomposition_method = self._DECOMPOSITION_CHOICES[(self.analysis_description["model_type"],self.analysis_description["decomposition"]["use_cupy"])]
		self.analysis_description["decomposition"]["method"]=decomposition_method

		method = getattr(self,decomposition_method)
		method()

	def apply_fix_W(self):
		if not self.W_fixed_bool is None:
			self.W[self.W_fixed_bool]=self.W_fixed_values[self.W_fixed_bool]









