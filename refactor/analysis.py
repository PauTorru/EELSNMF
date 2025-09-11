
import hyperspy.api as hs
import pandas as pd



default_q_methods = {
	("deltas","standard"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("convolved_single","standard"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	},
	("deltas","standard_cupy"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("convolved_single","standard_cupy"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
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

	def calculate_loadings(self):
		self.loadings = self.H.reshape([-1]+list(self.spatial_shape))




	def get_edge_from_component(self,component_id,edge):

		data = self.G[self._G_slices[edge]]@self.W[self._W_slices[edge],component_id]

		return hs.signals.Signal1D(data,axes=[self._eaxis_parameters])


	def quantify_components(self, method=None):

		if method is None:
			method = self._default_component_quantification_method(self)
		elif isinstance(method,str):
			method = getattr(self,method)

		array_component_quantification = method(self)

		self.component_quantification = pd.DataFrame(array_component_quantification,
			columns=["component_{}".format(i) for i in range(self.W.shape[1])],
			index = [i.split("_")[0] for i in self.edges])

		try:
			display(self.component_quantification)
		except:
			pass

		return array_component_quantification



	def _default_component_quantification_method(self):

		qm = default_q_methods[(self._built_G,self._decomposition_method)]["component_quantification"]

		return getattr(self,qm)

	def _default_spatial_quantification_method(self):

		qm = default_q_methods[(self._built_G,self._decomposition_method)]["spatial_quantification"]

		return getattr(self,qm)

	def get_chemical_maps(self,method=None,quantified=True):

		if method is None:
			method = self._default_spatial_quantification_method(self)

		array_spatial_quantification = method(self)

		if quantified:
			array_spatial_quantification*=100/array_spatial_quantification.sum(0)[np.newaxis,...]


		self.chemical_maps={}
		for el,array in zip(self.edges,array_spatial_quantification):
			self.chemical_maps[el]=array

		return self.chemical_maps






