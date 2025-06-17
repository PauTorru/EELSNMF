
import numpy as np
import tqdm
import os
import pickle as pkl
import copy
import matplotlib.pyplot as plt
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
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf
import pandas as pd


def load_decomposition(fname):
	with open(fname,"rb") as f:
		out = pkl.load(f)

	d,x = out.cl
	out.cl = hs.signals.Signal1D(out.cl[0])
	out.cl.axes_manager[-1].offset = x[0]
	out.cl.axes_manager[-1].scale = x[1]-x[0]

	d,x = out.ll
	out.ll = hs.signals.Signal1D(out.ll[0])
	out.ll.axes_manager[-1].offset = x[0]
	out.ll.axes_manager[-1].scale = x[1]-x[0]
	return out



def load(fname):
	s = hs.load(fname)
	if isinstance(s,hs.signals.Signal1D): # its either list or SI
		return s,None

	s = [i for i in l if isinstance(i,hs.signals.Signal1D)]

	if len(s)==2:
		ll = [i for i in s if any(i.axes_manager[-1].axis<0)][0]
		cl = [i for i in s if not any(i.axes_manager[-1].axis<0)][0]
		return cl,ll
	elif len(s)==1:# its either a core-loss or a lowloss
		s = s[0]
		if any(s.axes_manager[-1].axis<0):# is low loss
			return None,s
		else:
			return s,None
	else:
		raise Exception("File should contain either one (coreloss) or two (coreloss + lowloss) SIs. Insted there are {}".format(len(s)))




class EELSNMF:

	def __init__(self,core_loss, edges, low_loss= None,
														n_background = None,
														E0 = None,
														alpha = None,
														beta = None,
														fine_structure_ranges = None,
														ll_convolve = False,
														max_iters=100,
														tol=1e-5,
														xsection_type="Kohl"):

		"""
		Parameters
		----------

		core_loss: hs.signals.Signal1D or path
			Spectrum image to which the EELSNMF decomposition will be applied.

		edges: tuple, Eg. (O_K,Fe_L)
			Core-loss edges expected to be present in the spectrum image.

		low_loss: hs.signals.Signal1D or path
			Coacquired Low-loss. If supplied it will be used for convolution/deconvolution purposes.

		n_background: int
			Number of power-law backgrounds used in the model.

		E0, alpha, beta: float
			Acceleration voltage, convergence angle and collection angle of the acquistion.
			If None, it is read from image metadata.
			Units are respectively V, rad, and rad. (not kV, mrad)

		fine_structure_ranges: dict
			Dictionary specifying the range of fine structure for given edges.
			Eg. {"Fe_L":(708.,730.)}

		ll_convolve: bool or tuple
			If this variable is None or False, the G matrix will not be convolved with the low loss.
			If it is true, it will be convolved at each position with the corresponding low-loss spectrum in the low-loss SI.
			If it is a set of indices, all of the G matrix will be convolved with the low-loss spectrum at that position in the low-loss SI.
		
		max_iters: int
			Maximum iterations of the NMF algorithm

		tol: float
			convergence tolerance on the error.



		"""
		self.xsection_type=xsection_type
		self.tol=tol
		self.max_iters=max_iters
		self.ll_convolve = ll_convolve
		self.ll = None #default
		if isinstance(core_loss,str):
			cl,ll = load(core_loss)
			self.cl = cl
			self.ll = ll
		elif isinstance(core_loss,hs.signals.Signal1D):
			self.cl = core_loss
		else:
			raise Exception("Bad core_loss argument")

		if not low_loss is None:
			if isinstance(low_loss,str):
				cl,ll = load(low_loss)
				self.ll = ll
			elif isinstance(low_loss,hs.signals.Signal1D):
				self.ll = low_loss

			else:
				raise Exception("Bad low_loss argument")


		self.edges = edges

		if n_background is None:
			self.n_background = len(edges)

		else:
			self.n_background = n_background

		if E0 is None:
			self.E0 = self.cl.metadata.Acquisition_instrument.TEM.beam_energy*1e3 # in V
		else:
			self.E0 = E0

		if alpha is None:
			self.alpha = self.cl.metadata.Acquisition_instrument.TEM.convergence_angle*1e-3 # in rad
		else:
			self.alpha = alpha


		if beta is None:
			self.beta = self.clmetadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle*1e-3 #in rad
		else:
			self.beta = beta

		self.fine_structure_ranges = fine_structure_ranges

		self.energy_size = self.cl.data.shape[-1]


		self.build_G()


	def build_G(self):
		self.energy_axis=self.cl.axes_manager[-1].axis
		if len(self.cl.data.shape)==2:
			p,e=self.cl.data.shape
			hl = em.MultiSpectrum.from_numpy(self.cl.data.reshape((1,p,e)),self.cl.axes_manager[-1].axis)
		else:
			hl = em.MultiSpectrum.from_numpy(self.cl.data,self.cl.axes_manager[-1].axis)
		if self.ll:
			ll = em.MultiSpectrum.from_numpy(self.ll.data,self.ll.axes_manager[-1].axis)

		xs=[]

		for edge in self.edges:

			element,edge_type = edge.split("_")
			if self.xsection_type == "Zezhong":

				x = ZezhongCoreLossEdgeCombined(hl.get_spectrumshape(),
					1,self.E0,self.alpha,self.beta,element,edge_type)
			elif self.xsection_type == "Kohl":
				x = KohlLossEdgeCombined(hl.get_spectrumshape(),
					1,self.E0,self.alpha,self.beta,element,edge_type)
			else:
				print("cross section type \"{}\" does not exist".format(self.xsection_type))


			xs.append(x)

		ll_comp = MscatterFFT(ll.get_spectrumshape(),
			ll,True,"edge","constant")

		bg = LinearBG(hl.get_spectrumshape(),
			rlist=np.linspace(1,self.n_background,self.n_background))

		comp_list = [bg]+xs+[ll_comp]

		self.xsection_idx={}
		for i,edge in enumerate(self.edges):
			element,edge_type=edge.split("_")
			self.xsection_idx[edge]=i+self.n_background



		self.model = em.Model(hl.get_spectrumshape(),components=comp_list)
		self.em_fitter = LinearFitter(hl,self.model)

		if self.ll_convolve==True:
			#Full deconvolution
			print("Implemention pending")
			return
		elif self.ll_convolve==False:
			self.em_fitter.calculate_A_matrix()

			G = self.em_fitter.A_matrix.copy()

		elif isinstance(self.ll_convolve,tuple):
			self.em_fitter.calculate_A_matrix()
			self.em_fitter.model.components[-1].llspectrum.setcurrentspectrum(self.ll_convolve)
			G = self.em_fitter.convolute_A_matrix()

		else:
			print("BAD ll_convolve ARGUMENT")
			return



		ax = self.cl.axes_manager[-1]
		freeGs=[]
		print("Creating ELNES")
		ne=-1



		for k,v in self.fine_structure_ranges.items():
			ne+=1

			ii,ff = ax.value2index(v)
			l = ff-ii+1

			G[ii:ff,self.n_background+ne:]=0#ne is to only set to zero the corresponding edge

			if not self.ll_convolve:
				freeG = np.zeros((self.energy_size,l))
				for r,i in enumerate(range(ii,ff+1)):
					freeG[i,r]=1

			elif isinstance(self.ll_convolve,tuple):
				lldata = self.em_fitter.model.components[-1].llspectrum.data
				o = lldata.argmax()#ZLP position

				freeG = np.zeros((self.energy_size,l))
				for r,i in enumerate(range(ii,ff+1)):

					#put the lldata into freeG so that o coincides with i
					if i>o:
						freeG[i-o:,r]=lldata[:-(i-o)]
					elif i<o:
						freeG[:-(o-i),r]=lldata[o-i:]

					else:#i==o
						freeG[:,r]=lldata


			elif self.ll_convolve:
				#full convolution
				print("NOT implemented")
				return

			else:
				print("BAD ll_convolve")
				return



			freeGs.append(freeG)
		self._Gstructure = [G.shape[1]]+[i.shape[1] for i in freeGs]


		
		self._edge_slices={}
		self._W_edge_slices={}
		cs = np.cumsum(self._Gstructure)

		for i,k in enumerate(self.fine_structure_ranges.keys()):
			self._edge_slices[k]=np.s_[:,cs[i]:cs[i+1]] # this gives you the slice to get a given edge: G[slice]@W[slice,comp] for and edge of a component
			self._W_edge_slices[k]=np.s_[cs[i]:cs[i+1]]
	

		self.G = np.concatenate([G]+freeGs,axis=1)
		self.G[self.G<0]=0




	def decomposition(self,n_comps,W_init=None):
		self.n_comps=n_comps
		self.error_log=[]
		self.X = self.cl.data.reshape((-1,self.energy_size)).T
		#self.cl.decomposition()
		#GW = self.cl.get_decomposition_factors().data.T[:,:n_comps]
		#self.H = self.cl.get_decomposition_loadings().data[:n_comps].reshape((n_comps,-1))
		if not W_init is None:
			self.W = W_init
			self.H = np.abs(np.linalg.lstsq(self.G@self.W, self.X,rcond=None)[0])
		else:	
			GW,self.H = initialize_nmf(self.X,n_comps)
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])

		self.H[np.isnan(self.H)]=1e-10
		self.H[np.isinf(self.H)]=1e-10
		self.W[np.isnan(self.W)]=1e-10
		self.W[np.isinf(self.W)]=1e-10

		#fixed products
		self.GtX = self.G.T@self.X
		self.GtG = self.G.T@self.G


		error_0 = abs(self.X-self.G@self.W@self.H).sum()

		for i in range(1,self.max_iters+1):


		    #Update W
		    WHHt = self.W@self.H@self.H.T

		    num = self.GtX@self.H.T 
		    denum = self. GtG@WHHt

		    self.W*=num/denum

		    #update H
		    WH = self.W@self.H

		    num = self.W.T@self.GtX
		    denum = self.W.T@self.GtG@WH

		    self.H*=num/denum

		    error = abs(self.X-self.G@self.W@self.H).sum()
		    self.error_log.append(error)

		    if error_0-error<=self.tol and i>2:
		    	pass
		    	#print("Converged after {} iterations".format(i))
		    	#return

		    if i%50==0:
		    	print("Error = {} after {} iterations".format(error,i))
		    error_0 = error

		    #shifts to prevent 0 locking
		    self.W[self.W==0]=1e-10
		    self.H[self.H==0]=1e-10

		"""

		#gradient descent

		if not hasattr(self,"eta"):
			self.eta=1e-10

		for i in range(1,self.max_iters+1):


		    #Update W
		    WHHt = self.W@self.H@self.H.T

		    num = self.GtX@self.H.T 
		    denum = self.GtG@WHHt

		    self.W+=self.eta*(num-denum)

		    #update H
		    WH = self.W@self.H

		    num = self.W.T@self.GtX
		    denum = self.W.T@self.GtG@WH

		    self.H+=self.eta*(num-denum)



		    error = abs(self.X-self.G@self.W@self.H).sum()

		    if error_0-error<=self.tol and i>2:
		    	pass
		    	#print("Converged after {} iterations".format(i))
		    	#return

		    if i%50==0:
		    	print("Error = {} after {} iterations".format(error,i))
		    error_0 = error

		#update W
		"""
		

	def plot_factors(self):
		plt.figure("Factors")
		plt.clf()
		for i in range(self.n_comps):
			plt.plot(self.energy_axis,(self.G@self.W).T[i])

	def get_edge_from_component(self,component_id,edge):

		out = self.G[self._edge_slices[edge]]@self.W[self._W_edge_slices[edge],component_id]

		out = hs.signals.Signal1D(out)
		out.axes_manager[-1].offset = self.energy_axis[0]
		out.axes_manager[-1].scale = self.cl.axes_manager[-1].scale
		out.axes_manager[-1].name = self.cl.axes_manager[-1].name
		out.axes_manager[-1].units = self.cl.axes_manager[-1].units
		return out

	def save(self,fname):
		temp = self.cl.deepcopy()
		temp_ll = self.ll.deepcopy()
		self.cl = (self.cl.data,self.cl.axes_manager[-1].axis)
		self.ll = (self.ll.data,self.ll.axes_manager[-1].axis)

		with open(fname,"wb") as f:
			pkl.dump(self,f)

		self.cl= temp
		self.ll = temp_ll
		return

	def quantify_components(self):
		simplified_W=np.zeros((
			len(self.edges),
			self.W.shape[1]
			))
		for comp in range(self.W.shape[1]):
			for i,kv in enumerate(self.xsection_idx.items()):
				el,idx=kv
				simplified_W[i,comp]=self.W[idx,comp]


		simplified_W*=100/simplified_W.sum(0)[np.newaxis,:]

		self.quantification = pd.DataFrame(simplified_W,
			columns=["component_{}".format(i) for i in range(self.W.shape[1])],
			index = [i.split("_")[0] for i in self.xsection_idx.keys()])
		display(self.quantification)















