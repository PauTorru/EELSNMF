from .imports import *

def norm(x):
	"""

	Parameters
	----------
	x : array
		

	Returns
	-------
	array
		Array normalized from 0 to 1

	"""
	return (x-x.min())/(x.max()-x.min())


def find_2factors(n):
	"""
	Parameters
	----------
	n : int
		

	Returns
	-------
	tuple of ints
		Returns two integers (a,b) so that a*b=n

	"""
	
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
	"""
	Finds the indices of an array that containt the element closest to a value
	Parameters
	----------
	ax : array
		Array where to look for a given value
		
	v : float
		Value to look for in the array
		

	Returns
	-------
	int
		Index of the element closest to v in ax.

	"""
	if hasattr(v,"__iter__"):
		return [abs(ax-iv).argmin() for iv in v]
	else:
		return abs(ax-v).argmin()

def convolve(a,b):
	"""1-d convolution. the shape of a,b has to be even.

	Parameters
	----------
	a : np.array
		Typically it will be a column of the unconvolved G matrix.
	b : np.array
		Typically it will be a single low loss spectrum

	Returns
	-------

	array
		convolution of a with b.

	
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


def moving_average(a, n=3):
	"""

	Parameters
	----------
	a : array
		
	n : int
		 (Default value = 3)
		 Size of the moving arverage window

	Returns
	-------
	array

	"""
	n-=(1-n%2)
	ret = np.cumsum(a,axis=-1, dtype=float)
	ret[...,n:] = ret[...,n:] - ret[...,:-n]
	return np.pad(ret[...,n - 1:] / n,np.array([[0,0],[0,0],[(n-1)//2,(n-1)//2]]),mode="edge")

def all_arrays_equal(array_list):
	"""
	Check if all arrays in array_list are equal
	
	Parameters
	----------
	array_list : list of arrays
		

	Returns
	-------
	bool

	"""
	first_array = array_list[0]
	return all(np.array_equal(first_array, arr) for arr in array_list[1:])

def match_axis(s,new_axis):
	"""

	Parameters
	----------
	s : hs.signal.Signal1D
		
	new_axis : array
		the channels of the spectra will match this new_axis.
		

	Returns
	-------
	hs.signal.Signal1D
		Signal with the spectral axis matching new_axis

	"""

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
			interp = sc.interpolate.interp1d(xf,yf[xi],kind="cubic")
			out[xi,:]=interp(new_axis)
	
	if s.data.ndim==3:
		sx,sy,se = s.data.shape
		yf = s.data
		xf = s.axes_manager[-1].axis
	
		out = np.zeros ((sx,sy,new_axis.shape[0]))
	
		for xi in range(sx):
			for yi in range(sy):
				interp = sc.interpolate.interp1d(xf,yf[xi,yi],kind="cubic")
				out[xi,yi,:]=interp(new_axis)

	out =  hs.signals.Signal1D(out)
	out.axes_manager[-1].offset = offset
	out.axes_manager[-1].scale = scale

	return out

class ListOfSI():
	""" 
	Allows processing a list of spectrum images of different sizes as one single dataset.

	To do that, the spectra of the different spectrum images is matched to have the same "calibration" and
	the spatial dimensions are unfolded, keeping track of the original dimensions.

	Parameters
	----------
	slist : list of hs.signal.Signal1D
		
	enery_axis : None or array
		Axis to which to interpolate all SIs. If None it is assumed that all SI already have the same energy axis.
	
	"""
	def __init__(self,slist,energy_axis=None):
		
		self.len = len(slist)
		if all_arrays_equal([i.axes_manager[-1].axis for i in slist]):
			self.energy_axis = slist[0].axes_manager[-1].axis
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

		unfolded_data = np.zeros(((self.dim_list[:,0]*self.dim_list[:,1]).sum(),self.dim_list[0,-1]))

		filled=0
		self.spatial_dim_list=[]
		for s in self.slist:
			sx,sy,se = s.data.shape
			self.spatial_dim_list.append(sx*sy)
			unfolded_data[filled:filled+sx*sy,:]=s.data.reshape((-1,self.edim))
			filled+=sx*sy

		self.unfolded_si = hs.signals.Signal1D(unfolded_data)
		self.unfolded_si.axes_manager[-1].offset = self.energy_axis[0]
		self.unfolded_si.axes_manager[-1].scale = self.energy_axis[1]-self.energy_axis[0]
		return

	@property
	def unfolded_data(self):
		""" """
		return self.unfolded_si.data

	@unfolded_data.setter
	def unfolded_data(self,value):
		"""	"""
		self.unfolded_si.data=np.asarray(value)

	def fold_array(self,array):
		"""
		Folds array back to the dimension of the original set of spectrum images.
		The array can have the same spatial dimension (output will be flat 2D images)
		or the same spatial and spectral dimensions
		(output will be a list of array with the same dims as the original list of SIs).

		Parameters
		----------
		array : array
			

		Returns
		-------
		list of arrays

		"""
		out=[]
		filled=0
		for n,d in zip(self.spatial_dim_list,self.dim_list):
			folded_array = array[filled:filled+n,...].reshape(tuple(d[:-1])+(-1,))
			if folded_array.shape[-1]==1:
				folded_array = folded_array[...,0]
			out.append(folded_array)
			filled+=n
		return out


	def plot_decomposition_results(self,component,type="decomposition",figure = 0):
		"""
		Plots decomposition results with loadings in the same shape as original SIs.

		Parameters
		----------
		component : int
			Spectral component index.
			
		type : "decompositon" or "bss"
			 (Default value = "decomposition")
		figure : int or str
			 Name of the figure where the results will be plotted.
			 (Default value = 0)


		Returns
		-------
		None

		"""
		self.len+=1
		self.find_plot_structure()
		self.len-=1
		if type=="decomposition":
			l = self.unfolded_si.get_decomposition_loadings()
			f = self.unfolded_si.get_decomposition_factors()
		elif type=="bss":
			l = self.unfolded_si.get_bss_loadings()
			f = self.unfolded_si.get_bss_factors()
			
		plt.figure(figure)
		plt.clf()
		self.plot_array(l.data[component],False)
		r,c = self.plot_structure
		ax = plt.subplot(r,c,r*c)
		plt.plot(self.energy_axis,f.data[component])

	def plot_cluster_results(self):
		"""
		Plots cluster analysis results with labels in the same shape as original SIs.

		Returns
		-------
		None

		"""
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
		""" 
		Finds the optimal panel configuration for the results and stores it in self.plot_structure

		"""
		rows = int(np.sqrt(self.len))
		cols = self.len//rows
		while rows*cols!=self.len:
			rows-=1
			cols =self.len//rows
			if rows==0:
				raise Exception("failed to find good plot_structure, which should never happen")
		self.plot_structure =(rows,cols)
		return

	def plot_array(self,array,extra_row=False,vmin=None,vmax=None,cmap=None):
		"""
		Folds and plots an array in the shapes of the original SIs.

		Parameters
		----------
		array : array
			
		extra_row : bool
			Adds an extra row of panels in case extra info wants to be plotted to the figure.
			 (Default value = False)
		vmin : float
			Common vmin passed to plt.imshow for each panel.
			 (Default value = None)
		vmax : float
			Common vmin passed to plt.imshow for each panel.
			 (Default value = None)
		cmap : Matplotlib colormap
			 

		Returns
		-------
		plot figure

		"""
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
		return plt.gcf()

	def save(self,fname,overwrite=False):
		"""
		Saves ListOfSI object. It is saved as a folder with two files SI.hspy for the raw data
		and object.pkl for the rest of the information about the object. (Original SIs are not saved)

		Parameters
		----------
		fname : str
			
		overwrite : bool
			 (Default value = False)

		"""
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
	"""
	Loads ListOfSI object from path
	
	Parameters
	----------
	path : str or Path
		

	Returns
	-------
	ListOfSI

	"""
	si = hs.load(os.path.join(path,"SI.hspy"))
	with open(os.path.join(path,"object.pkl"),"rb") as f:
		out = pkl.load(f)

	out.unfolded_si=si

	return out

