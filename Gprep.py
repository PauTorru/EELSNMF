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

	self.xsection_idx={}
	for i,edge in enumerate(self.edges):
		element,edge_type=edge.split("_")
		self.xsection_idx[edge]=i+self.n_background
	##############################
	############################## 

	##############################
	###############################Handling of fine structure parameters in the G matrix
	self.ax = self.cl.axes_manager[-1]
	self.model = em.Model(hl.get_spectrumshape(),components=comp_list)
	self.em_fitter = LinearFitter(hl,self.model)


	self.em_fitter.calculate_A_matrix()
	G = self.em_fitter.A_matrix.copy()

	if self.ll_convolve==True:
		self._prepare_full_convolution_G(G)
		self._ucG,freeGs_shape  =self._prepare_dirac_G(G)#unconvolved G
		
		return

	elif isinstance(self.ll_convolve,tuple):
		self.G=self._prepare_single_spectrum_convolution_G(G)
		self._ucG,freeGs_shape =self._prepare_dirac_G(G)

	else:
		self.G,freeGs_shape=self._prepare_dirac_G(G)
		self._ucG=self.G.copy()
	##############################
	##############################

	self.G[self.G<0]=0
	self._Gstructure = [G.shape[1]]+freeGs_shape
	self._edge_slices={}
	self._W_edge_slices={}
	cs = np.cumsum(self._Gstructure)

	for i,k in enumerate(self.fine_structure_ranges.keys()):
		self._edge_slices[k]=np.s_[:,cs[i]:cs[i+1]] # this gives you the slice to get a given edge: G[slice]@W[slice,comp] for and edge of a component
		self._W_edge_slices[k]=np.s_[cs[i]:cs[i+1]]

	#set xsections to 0 on their corresponding fine structure range
	#this prevents issues when fitting the fine structure (dips below the xsection are not fitted)
	#Ideally sum of free parameters is linked to the cross section parameter but I could not get 
	for k in self.xsection_idx.keys():
		ii,ff = self.ax.value2index(self.fine_structure_ranges[k])
		self.G[ii:ff+2,self.xsection_idx[k]]=0 #ff+2 avoids overlap with fine structure?



def _prepare_dirac_G(self,G):
	freeGs=[]
	ne=-1
	for k,v in self.fine_structure_ranges.items():
		ne+=1

		ii,ff = self.ax.value2index(v)
		l = ff-ii#+1  #this overlaps with cropped xsection. problematic?
		freeG = np.zeros((self.energy_size,l))
		for r,i in enumerate(range(ii,ff)):#+1)):
			freeG[i,r]=1
		freeGs.append(freeG)
	out = np.concatenate([G]+freeGs,axis=1).astype(self.dtype)
	out[out<0] = 0
	return out,[i.shape[1] for i in freeGs]

	
def _prepare_single_spectrum_convolution_G(self,G):


	self.llspectrum=self.ll_data_flat[:,self._ll_id]
	self.llspectrum[self.llspectrum<0]=0
	self.llspectrum/=self.llspectrum.sum()

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
	###### /section is the convolution of the xsections #####

	# now we create fine structure elements
	freeGs=[]
	uc_freeGs=[]#unconvolved
	ne = -1
	for k,v in self.fine_structure_ranges.items():
		ne+=1 #count fine structure elements
		ii,ff = self.ax.value2index(v)
		l = ff-ii#+1  #this overlaps with cropped xsection. problematic?
		G[:ff,self.n_background+ne:]=0 #ne is to only set to zero the corresponding edge #[ii:ff,self.n_background+ne:]=0

		lldata = self.llspectrum
		o = lldata.argmax()#ZLP position

		freeG = np.zeros((self.energy_size,l))
		for r,i in enumerate(range(ii,ff)):#+1)):

			#put the lldata into freeG so that o coincides with i
			if i>o:
				freeG[i-o:,r]=lldata[:-(i-o)]
			elif i<o:
				freeG[:-(o-i),r]=lldata[o-i:]

			else:#i==o
				freeG[:,r]=lldata
		freeGs.append(freeG)
	
	return np.concatenate([G]+freeGs,axis=1).astype(self.dtype)


	

def _prepare_full_convolution_G(self,G0):


		#make low-loss data have the same spectral size as G

		if self.ll.data.shape[-1]>G0.shape[0]:
			self.ll_data_flat = self.ll_data_flat[:G0.shape[0],:]
		
		elif self.ll.data.shape[-1]<G0.shape[0]:
			missing = G0.shape[0]-self.ll_data_flat.shape[0]

			self.ll_data_flat = np.pad(self.ll_data_flat,((0,missing),(0,0)),mode="constant",constant_values=0)
		else:
			#already same shape
			pass

		#prep before G,GTG and GTX calculations
		self.X = self.cl.data.reshape((-1,self.energy_size)).T.astype(self.dtype)


		for p in range(self.ll_data_flat.shape[1]): #we'll make G at each position then calculate GTX[l,p] and GTG[l,l,p] and store that

			if p%int(self.ll_data_flat.shape[1]//10)==0:
				print("Preparing full convolution {}/{}".format(p,self.ll_data_flat.shape[1]))
			self._ll_id = p
			G=self._prepare_single_spectrum_convolution_G(G0)
			

			if not hasattr(self,"GtX"): # init here to no have to count final G size
				self.GtX = np.zeros([G.shape[1],self.X.shape[1]],dtype=self.dtype)

			if not hasattr(self,"GtG"): # init here to no have to count final G size
				self.GtG = np.zeros([G.shape[1],G.shape[1],self.X.shape[1]],dtype=self.dtype)


			self.GtX[:,p]=G.T@self.X[:,p]
			self.GtG[:,:,p]=G.T@G