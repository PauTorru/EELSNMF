
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
import scipy as sc


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

def find_2factors(n):
    rows = int(np.sqrt(n))+1
    cols = n//rows
    while rows*cols!=n:
        rows-=1
        cols = n//rows
        if rows==0:
            raise Exception("failed to find good plot_structure, which should never happen")
    return (rows,cols)



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
														xsection_type="Kohl",
														background_exps=None):

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

		xsection_type: str
			Wether to use "Kohl" or "Zezhong" cross section from pyEELSMODEL. "Kohl" uses the fast option.
		
		background_exps: tuple
			exponents for the power-law backgrounds




		"""
		self.xsection_type=xsection_type
		self.tol=tol
		self.max_iters=max_iters
		self.ll_convolve = ll_convolve
		self.background_exps = background_exps


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
		else:
			self.ll = None


		self.edges = edges

		if not self.background_exps is None:
			self.n_background = len(self.background_exps)
		elif n_background is None:
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

		

		if self.cl.data.shape[-1]%2==1 and self.ll_convolve!=False:
			self.cl=self.cl.isig[:-1] #make even for convolution.

		self.energy_size = self.cl.data.shape[-1]
		self.energy_axis = self.cl.axes_manager[-1].axis
		assert self.cl.data.min()>=0


		self.build_G()


	def build_G(self):

		############################## Create pyEELSMODEL SIs
		if len(self.cl.data.shape)==2:
			p,e=self.cl.data.shape
			hl = em.MultiSpectrum.from_numpy(self.cl.data.reshape((1,p,e)),self.cl.axes_manager[-1].axis)
		else:
			hl = em.MultiSpectrum.from_numpy(self.cl.data,self.cl.axes_manager[-1].axis)

		##############################
		############################## Create pyEELSMODEL xsections

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
		##############################
		############################## Create pyEELSMODEL background
		if not self.background_exps is None:
			bg = LinearBG(hl.get_spectrumshape(),
				rlist=self.background_exps)
		else:
			bg = LinearBG(hl.get_spectrumshape(),
				rlist=np.linspace(1,self.n_background,self.n_background))
		

		##############################
		############################## Create pyEELSMODEL low loss and background


		comp_list = [bg]+xs


		##############################
		############################## Keep track of idx for each edge for quantification afterwards
		self.xsection_idx={}
		for i,edge in enumerate(self.edges):
			element,edge_type=edge.split("_")
			self.xsection_idx[edge]=i+self.n_background
		##############################
		############################## 



		self.model = em.Model(hl.get_spectrumshape(),components=comp_list)
		self.em_fitter = LinearFitter(hl,self.model)


		self.em_fitter.calculate_A_matrix()
		G = self.em_fitter.A_matrix.copy()

		if self.ll_convolve==True:
			self._prepare_full_convolution(self,G)


		else:
			if isinstance(self.ll_convolve,tuple):
				#spectrum image or spectrum line handling
				assert len(self.ll_convolve)<=2
				if len(self.ll_convolve)==2: #SI
					i,j = self.ll_convolve
					self.llspectrum = self.ll.data[i,j]
				elif len(self.ll_convolve)==1: #Sline
					i = self.ll_convolve[0]
					self.llspectrum=ll.data[i]
				
				self.llspectrum[self.llspectrum<0]=0

				# convolution expects same spectral shape
				if self.llspectrum.shape[-1]>G.shape[0]:
					self.llspectrum=self.llspectrum[:G.shape[0]]
				elif self.llspectrum.shape[-1]<G.shape[0]:
					missing = G.shape[0]-self.llspectrum.shape[-1]
					self.llspectrum = np.pad(self.llspectrum,(0,missing),mode="constant",constant_values=(0,0))
				else:
					pass #already equal shape


				for i in range(self.n_background,G.shape[1]):
					G[:,i] = convolve(G[:,i],self.llspectrum)
				



			ax = self.cl.axes_manager[-1]
			freeGs=[]
			print("Creating ELNES")
			ne=-1



			for k,v in self.fine_structure_ranges.items():
				ne+=1

				ii,ff = ax.value2index(v)
				l = ff-ii+1
					
				if isinstance(self.ll_convolve,tuple):
					G[:ff,self.n_background+ne:]=0#ne is to only set to zero the corresponding edge #[ii:ff,self.n_background+ne:]=0

					lldata = self.llspectrum
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


				elif self.ll_convolve==False:
					freeG = np.zeros((self.energy_size,l))
					for r,i in enumerate(range(ii,ff+1)):
						freeG[i,r]=1

				else:
					print("BAD ll_convolve")
					return



				freeGs.append(freeG)

			self.G = np.concatenate([G]+freeGs,axis=1)
			self.G[self.G<0]=0

		self._Gstructure = [G.shape[1]]+[i.shape[1] for i in freeGs]


		
		self._edge_slices={}
		self._W_edge_slices={}
		cs = np.cumsum(self._Gstructure)

		for i,k in enumerate(self.fine_structure_ranges.keys()):
			self._edge_slices[k]=np.s_[:,cs[i]:cs[i+1]] # this gives you the slice to get a given edge: G[slice]@W[slice,comp] for and edge of a component
			self._W_edge_slices[k]=np.s_[cs[i]:cs[i+1]]
	

		


	def decomposition(self,n_comps,W_init=None):
		self.n_comps=n_comps
		self.error_log=[]
		if self.ll_convolve==True:
			return self._fullconv_decomposition(self,W_init=W_init)

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
		if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
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

	def calculate_loadings(self):
		self.loadings = self.H.reshape([-1]+list(self.cl.data.shape[:-1]))

	def plot_loadings(self):
		self.calculate_loadings()
		if not hasattr(self,"plot_structure"):
			self.plot_structure = find_2factors(self.n_comps)
		plt.figure("Loadings")
		plt.clf()
		r,c= self.plot_structure
		for i in range(self.n_comps):
			ax = plt.subplot(r,c,i+1)
			plt.imshow(self.loadings[i])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title("Loading {}".format(i))
		plt.tight_layout()



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

	

	def _prepare_full_convolution(self,G):
		pass

	def _fullconv_decomposition(self,W_init=None):
		pass















