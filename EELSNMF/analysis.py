from .imports import *



default_q_methods = {
	("deltas","_default_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("convolved_single","standard"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	},
	("deltas","_cupy_default_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("convolved_single","_cupy_default_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	},
	("deltas","_default:kl_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("deltas","_cupy_default_kl_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},

	###################################################### Not implemented:

	("built_in_sum_rule","standard"):{
	"component_quantification":"component_sumrule_q",
	"spatial_quantification":"spatial_sumrule_q"
	},
	("built_in_sum_rule","standard_cupy"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	},
	("full_conv","full_conv"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	}
}


class Analysis:
    """ Mixin class with chemical analysis functionalities for EELSNMF """

	def calculate_loadings(self):
		""" """
		self.loadings = self.H.reshape([-1]+list(self.spatial_shape))




	def get_edge_from_component(self,component_id,edge):
		"""
		Extracts the ELNES of edge from component #component_id.

		Parameters
		----------
		component_id : int
		    
		edge : str
			str defining the edge. E. g. "Fe_L"
		    

		Returns
		-------

		hs.signals.Signal1D

		"""

		data = self.G[:,self.model._edge_slices[edge]]@self.W[self.model._edge_slices[edge],component_id]

		return hs.signals.Signal1D(data,axes=[self._eaxis_parameters])


	def quantify_components(self, method=None):

		"""
		Get the chemical quantification of all components acoording to method
		
		Parameters
		----------
		method : str
			see valid methods at EELSNMF.analysis.default_q_methods
		     (Default value = None)

		Returns
		-------
		array

		"""

		if method is None:
			method = self._default_component_quantification_method()
		elif isinstance(method,str):
			method = getattr(self,method)

		array_component_quantification = method()

		self.component_quantification = pd.DataFrame(np.round(array_component_quantification,2),
			columns=["component_{}".format(i) for i in range(self.W.shape[1])],
			index = [i.split("_")[0] for i in self.edges])

		try:
			display(self.component_quantification)
		except:
			pass

		return array_component_quantification

	def component_standard_q(self):
		""" standard quantification of a spectral component"""
		l = len(self.edges)

		q = np.zeros((l,self.n_components))

		for ll,el in enumerate(self.edges):
			for k in range(self.n_components):
				q[ll,k] = self.W[self.model.xsection_idx[el],k]

		q*=100/q.sum(0)[np.newaxis,:]

		return q

	def spatial_standard_q(self):
		""" pixel per pixel standard quantification"""
		N = np.zeros((len(self.edges),self.H.shape[1]))
		i=0
		for k,v in self.model.xsection_idx.items():
			N[i,:] = self.W[v]@self.H
			i+=1
		return N.reshape((-1,)+self.spatial_shape)




	def _default_component_quantification_method(self):
		""" """

		qm = default_q_methods[(self.analysis_description["model_type"],self.analysis_description["decomposition"]["method"])]["component_quantification"]

		return getattr(self,qm)

	def _default_spatial_quantification_method(self):
		""" """

		qm = default_q_methods[(self.analysis_description["model_type"],self.analysis_description["decomposition"]["method"])]["spatial_quantification"]

		return getattr(self,qm)

	def get_chemical_maps(self,quantified=True,method=None):
		"""
		Calculates the abundance maps of all elements in the analysis.
		Parameters
		----------
		quantified : bool
			If true, normalizes the quantification to 100% at each pixel
		     (Default value = True)
		method : 
			Quantification method to use. If None , falls back to methods defined in .default_q_methods
		     (Default value = None)

		Returns
		-------
		dict
			Dictionary with the quantification array for each element.

		"""

		if method is None:
			method = self._default_spatial_quantification_method()

		array_spatial_quantification = method()

		if quantified:
			array_spatial_quantification*=100/array_spatial_quantification.sum(0)[np.newaxis,...]


		self.chemical_maps={}
		for el,array in zip(self.edges,array_spatial_quantification):
			self.chemical_maps[el]=array

		return self.chemical_maps






