from .imports import *



default_q_methods = {
	("deltas","default_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification": "spatial_standard_q"
	},
	("convolved_single","default_decomposition"):{
	"component_quantification":"component_standard_q",
	"spatial_quantification":"spatial_standard_q"
	},
	("deltas","default_kl_decomposition"):{
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

		return hs.signals.Signal1D(data,axes=[{k:v for k,v in self._eaxis_parameters.items() if k!="axis"}])


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
			method = getattr(self,"component_standard_q")
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
			method = getattr(self,"spatial_standard_q")

		array_spatial_quantification = method()

		if quantified:
			array_spatial_quantification*=100/array_spatial_quantification.sum(0)[np.newaxis,...]


		self.chemical_maps={}
		for el,array in zip(self.edges,array_spatial_quantification):
			self.chemical_maps[el]=array

		return self.chemical_maps

	def evaluate_comp_contributions(self,edge = None):
		"""
		Calculates the contribution of a component to the overall model in terms of three figures of merit: Leave-out loss, power and explained variance.

		Parameters
		----------
		Edge : None or one of self.edges
			If not None, the contributions relative to only that edge are considered
			 (Default value = None)

		Returns
		-------
		tuple
			constains three lists: leave out loss for every component, power of every component and explained variance of every component.

		"""		

		X = self.X
		model = self.get_model()
		normX = (X**2).sum()
		Emodel = X-model
		normEmodel = (Emodel**2).sum()

		R2 = 1-normEmodel/normX

		loo, power, ev = [],[],[] #leave-one-out,power,explained variance approx
		for k in range(self.n_components):
			if edge is None:
				wk = self.G @ self.W[:, k]
			else:
				idxs = self.model._edge_slices[edge]
				wk = self.G[:,idxs]@self.W[idxs,k]
			hk = self.H[k, :]

			wk_sq = np.sum(wk**2)
			hk_sq = np.sum(hk**2)
			comp_sq = wk_sq * hk_sq

			power.append(comp_sq/normX)

			interaction_X = (wk.T @ X) @ hk.T
			#Ek = normX+comp_sq-2*interaction_X 
			evk = (2*interaction_X-comp_sq)/normX  #expansion of (||X||-||X-Comp||)/||X||
			ev.append(evk)
			
			
			interaction_Emodel = wk.T @ Emodel @ hk.T
			Ck = (comp_sq+2*interaction_Emodel)/normX # expansion of ||Error_withoutK||-||Error_fullmodel||=||X-(Model-Comp)||-||X-Model||
			#is equal to power if the model is fully converged
			loo.append(Ck)

		loo,power,ev= [np.array(i) for i in [loo,power,ev]]
		print(f"Total explained : {R2}")
		fancy_summary = pd.DataFrame(data ={
			"Component":range(self.n_components),
			"Leave-Out Loss":loo,
			"Relative Leave-Out Loss (%)":100*loo/loo.sum(),
			"Power":power,
			"Relative Power (%)":power*100/power.sum(),
			"Explained Variance":ev,
			"Relative Explained Variance (%)":100*ev/ev.sum()	
		})
		try:
			display(fancy_summary)
		except NameError:
			print(fancy_summary)

		return loo,power,ev






