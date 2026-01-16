from .imports import *
import torch
import torch.nn as nn
import torch.optim as optim

def norm(x):
	return (x-x.min())/(x.max()-x.min())


def find_2factors(n):
	#for multipanel plots
	rows = int(np.sqrt(n))+1
	cols = n//rows
	while rows*cols!=n:
		rows-=1
		cols = n//rows
		if rows==0:
			raise Exception("failed to find good plot_structure, which should never happen")

	if (rows==1 or cols==1) and n>2:
			rows,cols = find_2factors(n+1)
	return (rows,cols)

def find_index(ax,v):
	if hasattr(v,"__iter__"):
		return [abs(ax-iv).argmin() for iv in v]
	else:
		return abs(ax-v).argmin()

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


def torch_deco(X, G, rank, max_iter=500, lr=1e-2, device="cpu",W_init=None,H_init=None,lambda_kurt=1e-3,scheduler_step=1000,gamma=0.8,l1_H=0,l1_W=0):
	"""
	Minimize ||X - G W H||_F^2 using Adam + Softplus parameterization.
	
	Args:
		X: (m x n) target matrix
		G: (m x p) fixed matrix
		rank: latent dimension (inner size of W, H)
		max_iter: number of optimization iterations
		lr: learning rate
		device: "cpu" or "cuda"
	
	Returns:
		W, H such that X ≈ G W H

	Example: W,H = factorization_with_fixed_G(deco.X,deco.G,5,max_iter=10000,lr=1e-1,
								 W_init = torch.from_numpy(deco.W).float(),H_init=torch.from_numpy(deco.H).float())
	deco.W=W.numpy()
	deco.H=H.numpy()
	"""
	X, G = torch.from_numpy(X).float(), torch.from_numpy(G).float()
	X, G = X.to(device), G.to(device)
	m, n = X.shape
	_, p = G.shape  # G is (m x p)

	# Raw parameters (unconstrained)
	# Initialize raw parameters
	if W_init is not None:
		# Invert softplus to get raw parameter values
		W = nn.Parameter(W_init.to(device))
	else:
		W = nn.Parameter(torch.randn(p, rank, device=device))
		
	if H_init is not None:
		H = nn.Parameter(H_init.to(device))
	else:
		H = nn.Parameter(torch.randn(rank, n, device=device))

	optimizer = optim.Adam([W, H], lr=lr)
	loss_fn = nn.MSELoss()
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=gamma)

	loss_total=[]
	#loss_kurt=[]
	loss_mse=[]
	loss_l1H=[]
	loss_l1W=[]

	for it in range(max_iter):
		optimizer.zero_grad()

		# Enforce non-negativity
		#W = torch.nn.functional.softplus(W_raw)
		#H = torch.nn.functional.softplus(H_raw)

		# Reconstruction: G W H
		X_hat = G @ W @ H

		# Loss
		mse_loss = loss_fn(X_hat, X)
		l1H = l1_H*H.abs().sum()
		l1W = l1_W*W.abs().sum()
		#H_pos = H.clamp(min=0)  # enforce non-negativity before regularizing
		#indep_penalty = kurtosis_loss(H_pos)
		loss = mse_loss+l1H+l1W #- lambda_kurt * indep_penalty

		loss_total.append(loss.item())
		#loss_kurt.append(- lambda_kurt * indep_penalty.item())
		loss_mse.append(mse_loss.item())
		loss_l1H.append(l1H.item())
		loss_l1W.append(l1W.item())

		# Backprop + update
		loss.backward()
		optimizer.step()
		scheduler.step()
		with torch.no_grad():
			W.clamp_(min=0)
			H.clamp_(min=0)

		if (it+1) % 100 == 0:
			print(f"Iter {it+1}, Loss={loss.item():.6f}")

	return W.detach(), H.detach(),loss_total,loss_mse,loss_l1H,loss_l1W#loss_kurt


def kurtosis_loss(H):
	"""
	Encourages rows of H to be non-Gaussian (independent).
	H: (k, n) tensor, rows = components
	"""
	Hc = H - H.mean(dim=1, keepdim=True)	 # center each row
	m2 = (Hc ** 2).mean(dim=1)			   # variance
	m4 = (Hc ** 4).mean(dim=1)			   # fourth moment
	kurt = m4 / (m2 ** 2 + 1e-8) - 3.0	   # excess kurtosis
	return -(kurt.abs()).sum()			   # maximize |kurtosis|

def all_arrays_equal(array_list):    
    first_array = array_list[0]
    return all(np.array_equal(first_array, arr) for arr in array_list[1:])

def match_axis(s,new_axis):

	assert isinstance(s,hs.signals.BaseSignal)
	assert isinstance(new_axis,np.ndarray)
	assert new_axis.ndim==1

	offset = new_axis[0]
	scale = new_axis[1]-new_axis[0]
	
	if s.data.ndim==1:
		se = s.data.shape
		yf = s.data
		xf = s.axes_manager[-1].axis
		
		out = np.zeros((new_axis.shape[0]))	

		interp = sc.interpolate.interp1d(xf,yf,kind="cubic")
		out[:]=interp(new_axis)
	
	if s.data.ndim==2:
		sx,se = s.data.shape
		yf = s.data
		xf = s.axes_manager[-1].axis
	
		out = np.zeros ((sx,new_axis.shape[0]))
	
		for xi in range(sx):
			interp = sci.interp1d(xf,yf[xi],kind="cubic")
			out[xi,:]=interp(new_axis)
	
	if s.data.ndim==3:
		sx,sy,se = s.data.shape
		yf = s.data
		xf = s.axes_manager[-1].axis
	
		out = np.zeros ((sx,sy,new_axis.shape[0]))
	
		for xi in range(sx):
			for yi in range(sy):
				interp = sci.interp1d(xf,yf[xi,yi],kind="cubic")
				out[xi,yi,:]=interp(new_axis)

	out =  hs.signals.Signal1D(out)
	out.axes_manager[-1].offset = offset
	out.axes_manager[-1].scale = scale

	return out

class ListOfSI():
	def __init__(self,slist,energy_axis=None):
		
		self.len = len(slist)
		if all_arrays_equal([i.axis_manager[-1].axis for i in slist]):
			self.eaxis = slist[0].axes_manager[-1].axis
			self.slist=slist
		else:
			matched_list = []
			for s in slist:
				matched = match_axis(s,energy_axis)
				matched_list.append(matched)
				self.energy_axis = energy_axis
				self.slist = matched_list

		self.find_plot_structure()
		self.dim_list = np.array([s.data.shape for s in self.slist])
		self.edim = self.dim_list[0,-1]
		assert all(self.dim_list[:,-1]==self.edim)

		self.unfolded_data = np.zeros(((self.dim_list[:,0]*self.dim_list[:,1]).sum(),self.dim_list[0,-1]))

		filled=0
		self.spatial_dim_list=[]
		for s in self.slist:
			sx,sy,se = s.data.shape
			self.spatial_dim_list.append(sx*sy)
			self.unfolded_data[filled:filled+sx*sy,:]=s.data.reshape((-1,self.edim))
			filled+=sx*sy

		self.unfolded_si = hs.signals.Signal1D(self.unfolded_data)
		self.unfolded_si.axes_manager[-1].offset = self.eaxis[0]
		self.unfolded_si.axes_manager[-1].scale = slist[0].axes_manager[-1].scale
		return

	def fold_array(self,array):
		out=[]
		filled=0
		for n,d in zip(self.spatial_dim_list,self.dim_list):
			out.append(array[filled:filled+n].reshape(d[:-1]))
			filled+=n
		return out

	def plot_decomposition_results(self,component,type="decomposition"):
		self.len+=1
		self.find_plot_structure()
		self.len-=1
		if type=="decomposition":
			l = self.unfolded_si.get_decomposition_loadings()
			f = self.unfolded_si.get_decomposition_factors()
		elif type=="bss":
			l = self.unfolded_si.get_bss_loadings()
			f = self.unfolded_si.get_bss_factors()
			
		plt.figure(0)
		plt.clf()
		self.plot_array(l.data[component],False)
		r,c = self.plot_structure
		ax = plt.subplot(r,c,r*c)
		plt.plot(f.data[component])

	def plot_cluster_results(self):
		self.len+=1
		self.find_plot_structure()
		self.len-=1
		
		l = self.unfolded_si.get_cluster_labels()
		f = self.unfolded_si.get_cluster_signals()
		n=l.data.shape[0]
		l = (l.data * (np.array(range(l.data.shape[0])))[:,np.newaxis]).sum(0)
	 
		plt.figure(0)
		plt.clf()
		cmap = mpl.colors.ListedColormap(sns.color_palette(cc.glasbey,n).as_hex())
		self.plot_array(l,False,cmap = cmap,vmin=0.,vmax=n-1)
		r,c = self.plot_structure
		ax = plt.subplot(r,c,r*c)
		for i in range(f.data.shape[0]):
			plt.plot(self.eaxis,f.data[i],color = sns.color_palette(cc.glasbey,n).as_hex()[i])


	def find_plot_structure(self):
		rows = int(np.sqrt(self.len))
		cols = self.len//rows
		while rows*cols!=self.len:
			rows-=1
			cols =self.len//rows
			if rows==0:
				raise Exception("failed to find good plot_structure, which should never happen")
		self.plot_structure =(rows,cols)
		return

	def plot_array(self,array,extra_row=True,vmin=None,vmax=None,cmap=None):
		ims = self.fold_array(array)
		r,c = self.plot_structure
		if extra_row:
			r+=1
		plt.clf()
		for i,im in enumerate(ims):
			ax = plt.subplot(r,c,i+1)
			if cmap:
				plt.imshow(im,vmin=vmin,vmax=vmax,cmap=cmap)
			else: 
				plt.imshow(im,vmin=vmin,vmax=vmax)
			#plt.colorbar()
			ax.set_xticks([])
			ax.set_yticks([])
		return

	def save(self,fname,overwrite=False):
		if os.path.exists(fname) and overwrite:
			sure = input("About to completely delete current {}, are you sure? (y,n)".format(fname))
			if sure.lower()=="y":
				shutil.rmtree(fname)

		os.mkdir(fname)
		temp = self.unfolded_si.deepcopy()
		self.unfolded_si.save(os.path.join(fname,"SI.hspy"))
		if hasattr(self,"list"):
			del self.list
		del self.unfolded_si

		with open(os.path.join(fname,"object.pkl"),"wb") as f:
			pkl.dump(self,f)

		self.unfolded_si= temp
		return

def load_ListOfSI(path):
	si = hs.load(os.path.join(path,"SI.hspy"))
	with open(os.path.join(path,"object.pkl"),"rb") as f:
		out = pkl.load(f)

	out.unfolded_si=si

	return out

