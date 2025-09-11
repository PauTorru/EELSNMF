
import numpy as np
import scipy as sc
import pyEELSMODEL.api as em
import pyEELSMODEL
from pyEELSMODEL.components.linear_background import LinearBG
from pyEELSMODEL.components.CLedge.zezhong_coreloss_edgecombined import ZezhongCoreLossEdgeCombined
from pyEELSMODEL.components.CLedge.kohl_coreloss_edgecombined import KohlLossEdgeCombined
from pyEELSMODEL.components.MScatter.mscatterfft import MscatterFFT
from pyEELSMODEL.components.MScatter.mscatter import Mscatter
from pyEELSMODEL.components.gdoslin import GDOSLin
from pyEELSMODEL.fitters.linear_fitter import LinearFitter
import hyperspy.api as hs
from .utils import *




def convolve(a,b):
	"""1-d convolution. the shape of a,b has to be even.

	Parameters
	----------

	a: np.array
		Typically it will be a column of the unconvolved G matrix.

	b: np.array
		Typically it will be a single low loss spectrum


	Returns
	-------

	C: np.array
		Typically the low-loss convolved G row.

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
	def __init__(self):
		pass

	def __getstate__(self):
		state = self.__dict__.copy()
		state["parent"] = None
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)

	#@property
	#def G(self):
	#	if hasattr(self,"G"):
	#		return self.G
	#	else:
	#		return None



class ModelG:
	def __getstate__(self):
		return self.__dict__

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Restore model -> parent link
		if hasattr(self, "model"):
			self.model.parent = self

	@property
	def available_models(self):
		available_models = ["deltas","convolved_single"]
		return available_models
	

	def build_G(self,
		low_loss= None,
		fine_structure_ranges = {},
		backgrounds = None,
		model_type = "deltas",
		xsection_type="Kohl",
		):

		"""
		Parameters
		----------
		low_loss: hs.signals.Signal1D or path
			Coacquired Low-loss. If supplied it will be used for convolution/deconvolution purposes.

		backgrounds: int or iterable
			Number of power-law backgrounds used in the model or their exponents if it is an iterable.

		fine_structure_ranges: dict
			Dictionary specifying the range of fine structure for given edges.
			Eg. {"Fe_L":(708.,730.)}

		model_type: one of self.available_models

		xsection_type: str
			Wether to use "Kohl" or "Zezhong" cross section from pyEELSMODEL. "Kohl" uses the fast option.

		"""


		if not low_loss is None:
			self.ll = low_loss
			self.Xll = low_loss.data.reshape(self.X.data.shape)#low_loss needs to have the same shape (spectrally too) as core_loss

		
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


		self.model = MODEL_REGISTRY[model_type](self)

	




	def get_background_xsections_from_pyEELS(self):

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
	def __init__(self,parent):
		pass

#class SumRule

#class FullConv







MODEL_REGISTRY = {
    "deltas": Deltas,
    "convolved_single": ConvolvedSingle
}