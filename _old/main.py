
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
import Gprep
from Gprep import convolve
import matplotlib as mpl
import importlib
importlib.reload(Gprep)#remove after debugging?
try:
	import cupy as cp
except:
	print("cupy not available")


def load_decomposition(fname):
	with open(fname,"rb") as f:
		out = pkl.load(f)

	d,x = out.cl
	out.cl = hs.signals.Signal1D(out.cl[0])
	out.cl.axes_manager[-1].offset = x[0]
	out.cl.axes_manager[-1].scale = x[1]-x[0]
	out.ax = out.cl.axes_manager[-1]

	d,x = out.ll
	out.ll = hs.signals.Signal1D(out.ll[0])
	out.ll.axes_manager[-1].offset = x[0]
	out.ll.axes_manager[-1].scale = x[1]-x[0]
	return out

def norm(x):
	return (x-x.min())/(x.max()-x.min())

def load(fname):
	s = hs.load(fname)
	if isinstance(s,hs.signals.Signal1D): # its either list or SI
		return s,None

	s = [i for i in s if isinstance(i,hs.signals.Signal1D)]

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
														fine_structure_ranges = {},
														ll_convolve = False,
														max_iters=100,
														xsection_type="Kohl",
														background_exps=None,
														tol = 1e-6,
														use_cupy=False,
														init_nmf=None,random_state_nmf=None):

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

		xsection_type: str
			Wether to use "Kohl" or "Zezhong" cross section from pyEELSMODEL. "Kohl" uses the fast option.
		
		background_exps: tuple
			exponents for the power-law backgrounds

		tol: float
			Sets conditions to stop decomposition process. If the relative change in error is below tol the process is stopped.

		use_cupy: bool
			Use cupy for GPU acceleration of the decomposition algorithm.


		"""
		self.xsection_type=xsection_type
		self.tol=tol
		self.max_iters=max_iters
		self.ll_convolve = ll_convolve
		self.background_exps = background_exps

		self.dtype =np.float64 #used for debugging memory issues and evaluate precision needed
		self.print_error_every=50
		self.use_cupy=use_cupy
		self.init_nmf=init_nmf
		self.random_state_nmf=random_state_nmf

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
			self.beta = self.cl.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle*1e-3 #in rad
		else:
			self.beta = beta

		self.fine_structure_ranges = fine_structure_ranges

		

		if self.cl.data.shape[-1]%2==1 and self.ll_convolve!=False:
			self.cl=self.cl.isig[:-1] #make even for convolution.

		self.energy_size = self.cl.data.shape[-1]
		self.energy_axis = self.cl.axes_manager[-1].axis
		assert self.cl.data.min()>=0

		if hasattr(self,"ll"):
			self.ll_data_flat = self.ll.data.reshape([-1]+[self.ll.data.shape[-1]]).T

		if isinstance(self.ll_convolve,tuple):
			self._ll_id=np.prod(self.ll_convolve)

		self.X = self.cl.data.reshape((-1,self.energy_size)).T.astype(self.dtype)
		self.build_G()

	build_G = Gprep.build_G
	_prepare_full_convolution_G = Gprep._prepare_full_convolution_G
	_prepare_single_spectrum_convolution_G = Gprep._prepare_single_spectrum_convolution_G
	_prepare_dirac_G = Gprep._prepare_dirac_G


	def _init_XWH(self,W_init=None):
		if not hasattr(self,"G") or self.G is None:
			self.G = self._ucG.copy() #initialize with uncovolved G
		if not hasattr(self,"X"):
			self.X = self.cl.data.reshape((-1,self.energy_size)).T.astype(self.dtype)
		#self.cl.decomposition()
		#GW = self.cl.get_decomposition_factors().data.T[:,:n_comps]
		#self.H = self.cl.get_decomposition_loadings().data[:n_comps].reshape((n_comps,-1))
		if not W_init is None:
			self.W = W_init
			self.H = np.abs(np.linalg.lstsq(self.G@self.W, self.X,rcond=None)[0])
		else:	
			GW,self.H = initialize_nmf(self.X,self.n_comps,init=self.init_nmf,random_state=self.random_state_nmf)
			self.W = np.abs(np.linalg.lstsq(self.G, GW,rcond=None)[0])

		self.H[np.isnan(self.H)]=1e-10
		self.H[np.isinf(self.H)]=1e-10
		self.W[np.isnan(self.W)]=1e-10
		self.W[np.isinf(self.W)]=1e-10

		self.H=self.H.astype(self.dtype)
		self.W=self.W.astype(self.dtype)


	def decomposition(self,n_comps,W_init=None,W_fixed_bool=None,W_fixed_values=None):
		self.n_comps=n_comps
		self.error_log=[]
		if self.ll_convolve==True:
			return self._fullconv_decomposition(W_init=W_init)

		self._init_XWH(W_init)

		#fixed products
		if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtX = self.G.T@self.X
			self.GtG = self.G.T@self.G

		if self.use_cupy:
			return self._cupy_decomposition(W_fixed_bool=W_fixed_bool,W_fixed_values=W_fixed_values)

		error_0 = abs(self.X-self.G@self.W@self.H).sum()

		for i in range(1,self.max_iters+1):


		    #Update W
		    WHHt = self.W@self.H@self.H.T

		    num = self.GtX@self.H.T 
		    denum = self. GtG@WHHt

		    self.W*=num/denum
		    if not W_fixed_bool is None:
		    	self.W[W_fixed_bool]=W_fixed_values

		    #update H
		    WH = self.W@self.H

		    num = self.W.T@self.GtX
		    denum = self.W.T@self.GtG@WH

		    self.H*=num/denum

		    error = abs(self.X-self.G@self.W@self.H).sum()
		    self.error_log.append(error)

		    if abs((error_0-error)/error_0)<=self.tol and i>2:
		    	print("Converged after {} iterations".format(i))
		    	return

		    if i%self.print_error_every==0:
		    	print("Error = {} after {} iterations. Relative change = {}".format(error,i,abs((error_0-error)/error_0)))
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
		
	def _cupy_decomposition(self,W_fixed_bool=None,W_fixed_values=None):
		
		self.GtX = cp.array(self.GtX)
		self.GtG = cp.array(self.GtG)
		self.X = cp.array(self.X) 
		self.W = cp.array(self.W)
		self.G = cp.array(self.G)
		self.H = cp.array(self.H)

		if not W_fixed_bool is None:
		    	self.W[W_fixed_bool]=W_fixed_values

		error_0 = float(cp.sum(cp.absolute(cp.subtract(self.X,
							cp.matmul(cp.matmul(self.G,self.W),self.H))))) #error_0 = abs(self.X-self.G@self.W@self.H).sum()

		for i in range(1,self.max_iters+1):


		    #Update W
		    WHHt = cp.matmul(cp.matmul(self.W,self.H),cp.transpose(self.H))#self.W@self.H@self.H.T

		    num = cp.matmul(self.GtX,cp.transpose(self.H))#self.GtX@self.H.T 
		    denum = cp.matmul(self.GtG,WHHt)#self. GtG@WHHt

		    self.W = cp.multiply(self.W,cp.divide(num,denum))#*=num/denum
		    if not W_fixed_bool is None:
		    	self.W[W_fixed_bool]=W_fixed_values

		    #update H
		    WH = cp.matmul(self.W,self.H)#self.W@self.H

		    num = cp.matmul(cp.transpose(self.W),self.GtX)#self.W.T@self.GtX
		    denum = cp.matmul(cp.matmul(cp.transpose(self.W),self.GtG),WH)#self.W.T@self.GtG@WH

		    self.H = cp.multiply(self.H,cp.divide(num,denum))#*=num/denum

		    error = float(cp.sum(cp.absolute(cp.subtract(self.X,
							cp.matmul(cp.matmul(self.G,self.W),self.H))))) #error = abs(self.X-self.G@self.W@self.H).sum()

		    self.error_log.append(error)

		    if abs((error_0-error)/error_0)<=self.tol and i>2:
		    	print("Converged after {} iterations".format(i))
		    	break

		    if i%self.print_error_every==0:
		    	print("Error = {} after {} iterations. Relative change = {}".format(error,i,abs((error_0-error)/error_0)))
		    error_0 = error

		    #shifts to prevent 0 locking
		    self.W[self.W==0]=1e-10
		    self.H[self.H==0]=1e-10
		    if not W_fixed_bool is None:
		    	self.W[W_fixed_bool]=W_fixed_values

		self.X = self.X.get()
		self.W = self.W.get()
		self.G = self.G.get()
		self.H = self.H.get()
		self.GtG  = self.GtG.get()
		self.GtX  = self.GtX.get()


	def plot_factors(self):
		plt.figure("Factors")
		plt.clf()
		for i in range(self.n_comps):
			plt.plot(self.energy_axis,(self.G@self.W).T[i],label="Component {}".format(i))
		plt.legend()

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
		self.cl = (self.cl.data,self.cl.axes_manager[-1].axis)

		if hasattr(self,"ax"):
			del self.ax
		

		if isinstance(self.ll,hs.signals.Signal1D):
			temp_ll = self.ll.deepcopy()
			self.ll = (self.ll.data,self.ll.axes_manager[-1].axis)
		else:
			temp_ll=None

		

		with open(fname,"wb") as f:
			pkl.dump(self,f)

		self.cl= temp
		self.ll = temp_ll
		self.ax = self.cl.axes_manager[-1]
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
		try:
			display(self.quantification)
		except:
			pass




	def _fullconv_decomposition(self,W_init=None):
		if not hasattr(self,"GtG"):
			raise Exception("was _prepare_full_deco not run?")

		self._init_XWH(W_init)
		if self.use_cupy:
			return self._fullconv_cupy_decomposition()

		#error_0 = abs(self.X-self._ucG@self.W@self.H).sum()  We have no explicit G stored to save memory

		for i in range(1,self.max_iters+1):


		    #Update W
		    #WHHt = self.W@self.H@self.H.T

		    num = self.GtX@self.H.T 
		    denum = np.einsum("mlp,lk,kp,pq->mq",self.GtG,self.W,self.H,self.H.T)#self.GtG@WHHt

		    self.W*=num/denum

		    #update H
		    #WH = self.W@self.H

		    num = self.W.T@self.GtX
		    denum = np.einsum("kl,lmp,mq,qp->kp",self.W.T,self.GtG,self.W,self.H)

		    self.H*=num/denum

		    #error = abs(self.X-self.G@self.W@self.H).sum()
		    #self.error_log.append(error)

		    #if error_0-error<=self.tol and i>2:
		    	#pass
		    	#print("Converged after {} iterations".format(i))
		    	#return

		    if i%50==0:
		    	print("Error calculation not implemented. iters = {}".format(i))#.format(error,i))
		    #error_0 = error

		    #shifts to prevent 0 locking
		    self.W[self.W==0]=1e-10
		    self.H[self.H==0]=1e-10

	def _fullconv_cupy_decomposition(self):
		self.GtX = cp.array(self.GtX)
		self.GtG = cp.array(self.GtG)
		self.X = cp.array(self.X) 
		self.W = cp.array(self.W)
		self.G = cp.array(self.G)
		self.H = cp.array(self.H)

		for i in range(1,self.max_iters+1):


			    #Update W
			    #WHHt = self.W@self.H@self.H.T

			    num = cp.matmul(self.GtX,cp.transpose(self.H))#self.GtX@self.H.T 
			    denum = cp.einsum("mlp,lk,kp,pq->mq",self.GtG,self.W,self.H,cp.transpose(self.H))#self.GtG@WHHt

			    self.W = cp.multiply(self.W,cp.divide(num,denum))#*=num/denum

			    #update H
			    #WH = self.W@self.H

			    num = cp.matmul(cp.transpose(self.W),self.GtX)#self.W.T@self.GtX
			    denum = cp.einsum("kl,lmp,mq,qp->kp",self.W.T,self.GtG,self.W,self.H)

			    self.H = cp.multiply(self.H,cp.divide(num,denum))#*=num/denum

			    #error = abs(self.X-self.G@self.W@self.H).sum()
			    #self.error_log.append(error)

			    #if error_0-error<=self.tol and i>2:
			    	#pass
			    	#print("Converged after {} iterations".format(i))
			    	#return

			    if i%50==0:
			    	print("Error calculation not implemented. iters = {}".format(i))#.format(error,i))
			    #error_0 = error

			    #shifts to prevent 0 locking
			    self.W[self.W==0]=1e-10
			    self.H[self.H==0]=1e-10

		self.X = self.X.get()
		self.W = self.W.get()
		self.G = self.G.get()
		self.H = self.H.get()
		#self.GtG  = self.GtG.get()
		#self.GtX  = self.GtX.get()

	def plot_edges(self,normalize=False):
		plt.figure("Edges")
		plt.clf()
		r,c=find_2factors(len(self.fine_structure_ranges.keys()))
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			for i in range(self.n_comps):
				ii,ff=self.ax.value2index(self.fine_structure_ranges[edge])
				if normalize:
					plt.plot(self.energy_axis[ii:ff],norm(self.get_edge_from_component(i,edge).data[ii:ff]))
				else:
					plt.plot(self.energy_axis[ii:ff],self.get_edge_from_component(i,edge).data[ii:ff])

	def get_chemical_maps(self):
		self.chemical_maps={}
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):

			self.chemical_maps[edge]=(self.W[self.xsection_idx[edge],:]@self.H).reshape(self.cl.data.shape[:-1])
		return self.chemical_maps

	def get_quantified_chemical_maps(self):
		chemmaps = self.get_chemical_maps()
		qmaps=[]
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):
			qmaps.append(chemmaps[edge])
		qmaps=np.array(qmaps)
		qmaps/=qmaps.sum(0)[np.newaxis,...]
		qmaps*=100

		self.qmaps={}
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):
			self.qmaps[edge]=qmaps[ik]
		return self.qmaps

	def plot_quantified_chemical_maps(self):
		
		qmaps=self.get_quantified_chemical_maps()
		plt.figure("Quantified Chemical Maps")
		plt.clf()
		r,c=find_2factors(len(self.fine_structure_ranges.keys()))
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			plt.imshow(qmaps[edge])
			plt.colorbar()
		

	def plot_chemical_maps(self):
		plt.figure("Chemical Maps")
		plt.clf()
		r,c=find_2factors(len(self.fine_structure_ranges.keys()))
		chemmaps = self.get_chemical_maps()
		for ik,edge in enumerate(self.fine_structure_ranges.keys()):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			plt.imshow(chemmaps[edge])
			plt.colorbar()

	def plot_average_model(self):
		plt.figure("Model")
		plt.clf()
		plt.plot(self.energy_axis,self.X.mean(1),label="Data")
		plt.plot(self.energy_axis,(self.G@self.W@self.H).mean(1),label="Model")
		plt.legend()
	
	def plot_energy_ranges(self):
		plt.figure("Edges energy ranges")
		plt.clf()
		plt.plot(self.energy_axis,norm(self.X.mean(1)))
		ax=plt.gca()
		colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
		c=-1
		for k,v in self.fine_structure_ranges.items():
			c+=1
			ii,ff = v
			rect = mpl.patches.Rectangle((ii, 0.), ff-ii, 1,
                         alpha=0.5,label=k,color=colors[c])
			ax.add_patch(rect)
		ax.set_xlim(self.energy_axis.min(),self.energy_axis.max())
		ax.set_ylim(0,1)
		ax.set_yticks([])
		plt.legend()


	




















