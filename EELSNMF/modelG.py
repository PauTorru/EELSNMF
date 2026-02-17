from .imports import *
from .utils import *




def convolve(a,b):
	"""1-d convolution. The shape of a,b has to be even.

	Parameters
	----------
	a : np.array
		Typically it will be a column of the unconvolved G matrix.
	b : np.array
		Typically it will be a single low loss spectrum

	Returns
	-------
	array
	
	"""

	assert a.shape==b.shape
	assert a.shape[0]%2==0
	assert b.shape[0]%2==0
	assert len(a.shape)==1
	assert len(b.shape)==1

	d = a.shape[0]

	o = b.argmax()
	a_pad = np.pad(a,(d,d),mode="edge")
	b_pad = np.pad(b,(0,o),mode="edge")
	b_pad/=b_pad.sum()
	conv = sc.signal.fftconvolve(a_pad,b_pad,mode="valid",axes=-1)[:d]
	return conv



class BaseModel:
	""" """
	def __init__(self):
		self._G_rescaled_to1 = False
		pass

	def __getstate__(self):
		state = self.__dict__.copy()
		state["parent"] = None
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)


	def _rescale(self):
		"""Needed to enforce smoothness between fine structure and xsections, applied before decomposition"""
		if not self._G_rescaled_to1:
			self._scales = self.G.sum(0)
			self.G[:,:] /= self._scales[None,:]
			self._G_rescaled_to1 = True
			if hasattr(self.parent,"W"):
				self.parent.W*=self._scales[:,None]
		else:
			pass
		#for edge in self.parent.edges:
			
		#	idx = self.xsection_idx[edge]			
		#	self._edge_scales[edge] = self.G[(self.G[:,idx]>0).argmax(),idx] #first nonzero
		#	self.G[:,idx]/=self._edge_scales[edge]

		#self._G_rescaled_to1 = True

		return

	def _undo_rescale(self):

		if self._G_rescaled_to1:
			self.G[:,:] *= self._scales[None,:]
			self.parent.W[:,:] /= self._scales[:,None]
			self._G_rescaled_to1 = False
		else:
			pass
		"""Undo scaling after decomposition"""

		#for edge in self.parent.edges:
			
		#	idx = self.xsection_idx[edge]			
			
		#	self.G[:,idx]*=self._edge_scales[edge]
		#	if hasattr(self.parent,"W"):
		#		self.parent.W[idx,:]/=self._edge_scales[edge]

		#self._G_rescaled_to1 = False

		return




	#@property
	#def G(self):
	#	if hasattr(self,"G"):
	#		return self.G
	#	else:
	#		return None



class ModelG:
	""" Mixin class for the modelling of the G matrix in EELSNMF objects"""
	def __getstate__(self):
		return self.__dict__

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Restore model -> parent link
		if hasattr(self, "model"):
			self.model.parent = self

	@property
	def available_models(self):
		""" """
		available_models = ["deltas","convolved_single"]
		return available_models
	

	def build_G(self,
		low_loss= None,
		fine_structure_ranges = {},
		backgrounds = None,
		model_type = "deltas",
		xsection_type="Kohl",
		**kwargs):

		"""
		Builds the G matrix according to model_type.

		Parameters
		----------
		low_loss : hs.signal.Signal1d
			Low-loss corresponding to the same region of main signal.
			Can be a single spectrum or a spectrum image
			 (Default value = None)

		fine_structure_ranges : dict
			Dictionary of ELNES ranges for each edge. E.g. {"O_K":(525.,540.),"Fe_L":(705.,750.)}
			 (Default value = {})
		backgrounds : array
			Exponents of the power-laws to be used for fitting the background.
			 (Default value = None)
		model_type : one of EELSNMF.modelG.MODEL_REGISTRY
			 (Default value = "deltas")
		xsection_type : "Kohl" or "Zezhong"
			Cross-section type used from the ones available in pyEELSMODEL
			 (Default value = "Kohl")

		**kwargs :
			Passed to the specific model
		
		"""


		if not low_loss is None:
			self.ll = low_loss
			self.Xll = low_loss.data.reshape(self.X.data.shape)#low_loss needs to have the same shape (spectrally too) as core_loss
		else:
			self.ll = None

		
		self.fine_structure_ranges=fine_structure_ranges
		self.edges = list(self.fine_structure_ranges.keys())


		if backgrounds is None:
			self.backgrounds = np.linspace(1,len(self.edges),len(self.edges))
		elif isinstance(backgrounds,int):
			self.backgrounds = np.linspace(1,backgrounds,backgrounds)
		elif hasattr(backgrounds,"__iter__"):
			self.backgrounds = backgrounds
		else:
			raise Exception("Background argument invalid")

		self.n_background = len(self.backgrounds)
		self.xsection_type = xsection_type


		self._G0 = self.get_background_xsections_from_pyEELS()


		
		self.analysis_description["model_type"]=model_type


		self.model = MODEL_REGISTRY[model_type](self,**kwargs)

	




	def get_background_xsections_from_pyEELS(self):
		""" """

		hl = em.MultiSpectrum.from_numpy(self.X.T[np.newaxis,...],self.energy_axis)


		xs=[]
		for edge in self.edges:
			element,edge_type = edge.split("_")
			if self.xsection_type == "Zezhong":
				x = ZezhongCoreLossEdgeCombined(hl.get_spectrumshape(),
					1,self.E0,self.alpha,self.beta,element,edge_type)
			elif self.xsection_type == "Kohl":
				x = KohlLossEdgeCombined(hl.get_spectrumshape(),
					1,self.E0,self.alpha,self.beta,element,edge_type,fast=True)
			else:
				print("cross section type \"{}\" does not exist".format(self.xsection_type))
			xs.append(x)

		self.xsections = np.array(xs)

		bg = LinearBG(hl.get_spectrumshape(),rlist=self.backgrounds)



		comp_list = [bg]+xs

		self.em_model = em.Model(hl.get_spectrumshape(),components=comp_list)
		self.em_fitter = LinearFitter(hl,self.em_model)

		self.em_fitter.calculate_A_matrix()

		G0 = self.em_fitter.A_matrix.copy()

		return G0


class Deltas(BaseModel):
	"""Model where the ELNES for each element is modelled a series of dirac deltas (thereby making it completely free)."""

	def __init__(self,parent):
		super().__init__()
		self.parent=parent


		I = np.eye(self.parent.energy_size)
		Gf =[]
		Gf_sizes =[]

		for k,v in self.parent.fine_structure_ranges.items():

			ii,ff = find_index(self.parent.energy_axis,v)
			Gf.append(I[:,ii:ff])
			Gf_sizes.append(Gf[-1].shape[-1])


		Gf = np.concatenate(Gf,axis=1)

		
		self.Gf_sizes=Gf_sizes
		self.Gf = Gf
		self._G_structure = [self.parent._G0.shape[1]]+self.Gf_sizes

		self._edge_slices = { k:np.s_[i:f] for k,i,f in zip(self.parent.edges,np.cumsum(self._G_structure)[:-1],np.cumsum(self._G_structure)[1:])}

		self.G = np.concatenate([self.parent._G0,self.Gf],axis=1)

		self.xsection_idx = {}
		for i,l in enumerate(self.parent.edges):
			self.xsection_idx[l] = self.parent.n_background+i


		for k,v in self.parent.fine_structure_ranges.items():

			ii,ff = find_index(self.parent.energy_axis,v)
			l = self.xsection_idx[k]

			self.G[ii:ff,l] = 0 #cropping crossection


class ConvolvedSingle(BaseModel):
	"""Model where the ELNES for each element is modelled a series of dirac deltas (thereby making it completely free)."""
	def __init__(self,parent,low_loss_spectrum):
		super.__init__()
		self.parent=parent
		assert len(low_loss_spectrum.data.shape)==1 #single spectrum, not SI
		self.llspectrum_data = low_loss_spectrum.data
		self.llaxis = self.low_loss_spectrum.axes_manager[-1].axis

		# convolution expects same spectral shape
		if self.llspectrum_data.shape[-1]>G.shape[0]:
			self.llspectrum_data=self.llspectrum_data[:G.shape[0]]
		elif self.llspectrum_data.shape[-1]<G.shape[0]:
			missing = G.shape[0]-self.llspectrum_data.shape[-1]
			self.llspectrum_data = np.pad(self.llspectrum_data,(0,missing),mode="constant",constant_values=(0,0))
		else:
			pass


		G0_convolved = self.parent._G0.copy()

		for i in range(self.n_background,G0_convolved.shape[1]): #convolve xsections, not backgrounds
			G0_convolved[:,i] = convolve(self.parent._G0[:,i],self.llspectrum_data)

		#set xsections to 0 on the fine structure ranges:
		self.xsection_idx = {}
		for i,l in enumerate(self.parent.edges):
			self.xsection_idx[l] = self.parent.n_background+i


		for k,v in self.parent.fine_structure_ranges.items():

			ii,ff = find_index(self.parent.energy_axis,v)
			l = self.xsection_idx[k]

			G0_convolved[ii:ff,l] = 0 


		#fine structure elements: shifted low-loss
		freeGs=[]
		freeGs_sizes=[]
		o = self.llspectrum_data.argmax()
		for k,v in self.fine_structure_ranges.items():
			ii,ff = self.ax.value2index(v)
			l = ff-ii+1  #this overlaps with cropped xsection. problematic?
			freeG = np.zeros((self.energy_size,l))
			freeGs_sizes.append(l)
			for r,i in enumerate(range(ii,ff+1)):

				#put the lldata into freeG so that o coincides with i
				if i>o:
					freeG[i-o:,r]=self.llspectrum_data[:-(i-o)]/self.llspectrum_data[:-(i-o)].sum()
				elif i<o:
					freeG[:-(o-i),r]=self.llspectrum_data[o-i:]/self.llspectrum_data[o-i:].sum()

				else:#i==o
					freeG[:,r]=self.llspectrum_data/self.llspectrum_data.sum()
			freeGs.append(freeG)

		self.G=np.concatenate([G0_convolved]+freeGs,axis=1).astype(self.dtype)
		self.Gf_sizes=freeGs_sizes
		self.Gf = np.concatenate(freeGs,axis=1).astype(self.dtype)
		self._G_structure = [self.parent._G0.shape[1]]+self.Gf_sizes
		self._edge_slices = { k:np.s_[i:f] for k,i,f in zip(self.parent.edges,np.cumsum(self._G_structure)[:-1],np.cumsum(self._G_structure)[1:])}















#class SumRule

#class FullConv







MODEL_REGISTRY = {
	"deltas": Deltas,
	"convolved_single": ConvolvedSingle
}