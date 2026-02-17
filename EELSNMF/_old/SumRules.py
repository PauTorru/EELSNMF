
import numpy as np
from types import SimpleNamespace
import tqdm
import os
import pickle as pkl
import copy
import pandas as pd
import scipy as sc
import matplotlib as mpl
import importlib
#from . import main as enmf #EELSNMF as enmf
#importlib.reload(Gprep)#remove after debugging?
from sklearn.decomposition._nmf import _initialize_nmf as initialize_nmf
try:
	import cupy as cp
	CUPY_AVAILABLE = True
except:
	print("cupy not available")
	CUPY_AVAILABLE = False

#hard typed constants:
gT2={300e3:489.1*1e3,200e3:343.8*1e3,80e3:149.2*1e3}

def theta_E(e,kV):
    return e/gT2[kV]

def Factor(th,alpha,beta):
    
    out = []
    for i,t in enumerate(th):
        if t<=abs(alpha-beta):
            f = min(1,(beta**2)/(alpha**2))
        else:
            x = (alpha**2+t**2-beta**2)/(2*alpha*t)
            y = (beta**2+t**2-alpha**2)/(2*beta*t)
            sqrt = np.sqrt(4*(alpha**2)*(beta**2)+(alpha**2+beta**2-t**2)**2)
            f = (1/np.pi)*(np.arccos(x)+((beta**2)/(alpha**2))*np.arccos(y)-(1/(2*alpha**2))*sqrt)
        out.append(f)
    return np.array(out)
    
def convergent_psi(e,alpha,beta,kV=300e3,n_points=1000):
    assert alpha>0
    assert beta>0
    thE = theta_E(e,kV)
    int_dth = integral_over_th(thE,alpha,beta,n_points)
    return e/int_dth

def integral_over_th(thE,alpha,beta,n_points):
    th = np.linspace(0,alpha+beta,n_points)
    dth = th[1]-th[0]
    Fth=Factor(th,alpha,beta)
    integrand = (Fth*2*np.pi*th*dth)[:,np.newaxis]/(th[:,np.newaxis]**2+thE[np.newaxis,:]**2)

    return integrand.sum(0)
    
    

def psi(e,beta,kV=300e3):
    return e/np.log(1+(beta**2)/(theta_E(e,kV)**2))



def decomposition_SumRule(self,n_comps, convergent_probe_correction = False):

	#fixed products
		if not hasattr(self,"GtX") and not hasattr(self,"GtG"): # in case of full deconvolution they are already created
			self.GtX = self.G.T@self.X
			self.GtG = self.G.T@self.G

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
########################################################################################### the percalaco begins



def adapt_G_SumRule(self, convergent_probe_correction = False,n_points=1000):

	sigma = np.array([self.em_fitter.A_matrix[:,v] for k,v in self.xsection_idx.items()])
	list_emax = [self.ax.value2index(v[1]) for k,v in self.fine_structure_ranges.items()]
	Gs = sigma.copy()
	for i,ff in list_emax:
		Gs[i,:ff]=0

	
	if convergent_probe_correction:
		pp = convergent_psi(self.energy_axis,self.alpha,self.beta,self.E0,n_points=n_points)
	else:
		pp = psi(self.energy_axis,self.beta,self.E0)


	Gf = self.G[:,self._Gstructure[0]:]

	Bf_values = pp@Gf
	els = sigma.shape[0]
	f_size = Gf.shape[1]
	Bf_lengths = deco._Gstructure[1:]

	Bf = np.zeros((els,f_size))

	ii=0
	for el,b in enumerate(Bf_lengths):
		Bf[el,ii:ii+b]=Bf_values[ii:ii+b]
		ii+=b


	A = np.array([pp[:emax]@ga[:emax] for ga,emax in zip(sigma,list_emax)])
	
	K = np.diag(1/np.array(A))

	self.sum_rules=SimpleNamespace()

	self.sum_rules.K=K
	self.sum_rules.Bf=Bf
	self.sum_rules.Gs=Gs
	self.sum_rules.Gf=Gf
	
	self.sum_rules.W,self.sum_rules.H = init_XWH_sumrules(self)

	return



def init_XWH_sumrules(self):
	#if not hasattr(self,"G") or self.G is None:
	#		self.G = self._ucG.copy() #initialize with uncovolved G
		if not hasattr(self,"X"):
			self.X = self.cl.data.reshape((-1,self.energy_size)).T.astype(self.dtype)
		#self.cl.decomposition()
		#GW = self.cl.get_decomposition_factors().data.T[:,:n_comps]
		#self.H = self.cl.get_decomposition_loadings().data[:n_comps].reshape((n_comps,-1))
	#	if not W_init is None:
	#		self.W = W_init
	#		self.H = np.abs(np.linalg.lstsq(self.G@self.W, self.X,rcond=None)[0])
	#	else:	
		GW,self.sum_rules.H = initialize_nmf(self.X,self.n_comps,init=self.init_nmf,random_state=self.random_state_nmf)
		self.sum_rules.W = np.abs(np.linalg.lstsq(self.sum_rules.Gf+self.sum_rules.Gs@self.sum_rules.K@self.sum_rules.Bf, GW,rcond=None)[0])

		self.sum_rules.H[np.isnan(self.H)]=1e-10
		self.sum_rules.H[np.isinf(self.H)]=1e-10
		self.sum_rules.W[np.isnan(self.W)]=1e-10
		self.sum_rules.W[np.isinf(self.W)]=1e-10

		self.sum_rules.H=self.H.astype(self.dtype)
		self.sum_rules.W=self.W.astype(self.dtype)










