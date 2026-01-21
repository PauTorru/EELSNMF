from ..imports import *
from ..utils import moving_average, norm

try:
    from whittaker_eilers import WhittakerSmoother
except ImportError:
    raise ImportError(
        "WhittakerSmoother is required for EML processing. "
        "Please install it using: pip install 'EELSNMF[EML]'"
    )

def we_smooth(array,lmbda=1e2,order=2):
	whittaker_smoother = WhittakerSmoother(lmbda=lmbda, order=order, data_length = array.shape[-1])
	smoothed_data = whittaker_smoother.smooth(array)
	return np.array(smoothed_data)


class EML_Processing():
	"""
	Class to manage the analysis of data from EML lab.

	"""
	def __init__(self,sh,sl,**kwargs):
		"""Parameters
		---------------
		sh : Core-loss spectrum image
		sl : Low-Loss spectrum image
		params : Parameters for pyEELSMODEL """
		self.core_loss = sh
		self.low_loss = sl

		# Default parameters
		self.n_bins=1000
		self.despiked = False
		self.pre_e=3
		self.ewidth=50
		self.degree=100
		self.long_despike=10
		self.interpolationtype="cubic"
		self.fit_method="ols"
		self.n_background=3

		self.zlp_aligned = False
		self.despiked = False
		self.denoise_applied = False
		self.deconvolution_applied = False
		self.deconvolved_edge=[]

		for key,value in kwargs.items():
			setattr(self,key,value)

	def save_results(self,dirname,metadata_average_s=None):

		os.makedirs(dirname,exist_ok=True)


		if self.despiked:
			self.core_loss_despiked.save(os.path.join(dirname,"core_loss_despiked.hspy"),overwrite = True)

		if self.deconvolution_applied:
			for i in self.deconvolved_edge:
				i.save(os.path.join(dirname,"edge_deconvolved_{}.hspy".format(i.process_save_label)),overwrite = True)

		if hasattr(self,"mask_vacuum"):
			np.save(os.path.join(dirname,"mask_vacuum.npy"),self.mask_vacuum)
			for i in self.deconvolved_edge:
				self.average_spectrum_excluding_vacuum(i,self.mask_vacuum)
				self.average_spectrum_excluding_vacuum_signal.metadata.Analysis=str(metadata_average_s)
				self.average_spectrum_excluding_vacuum_signal.save(os.path.join(dirname,"average_spectrum_no_vacuum{}.hspy".format(i.process_save_label)),overwrite = True)

		with open(os.path.join(dirname,"pyEELSMODEL_fit.pkl"),"wb") as f:
			pkl.dump(self.fit,f)


	def zlp_align(self):


		self.low_loss.align_zero_loss_peak(also_align=[self.core_loss])
		if self.low_loss.data.shape[-1]%2==1:# pyEELSMODEL has problem with spectra with odd number of channels ((internal anger))
			self.low_loss=self.low_loss.isig[:-1]
			self.core_loss=self.core_loss.isig[:self.low_loss.data.shape[-1]]
		self.zlp_aligned=True

	def despike(self,si,threshold="auto"):
		self.core_loss_despiked = si.deepcopy()
		
		if threshold == "auto":

			#dif = np.diff(si.data)
			fomd = fom(si.data+100)#(abs(dif)/si.data.sum(-1)[:,:,np.newaxis])
			#fom/=fom.sum(-1)[:,:,np.newaxis]

			fomd = np.nan_to_num(fomd,nan=0.,posinf=0.,neginf=0.)
			his,bins = np.histogram(fomd.ravel(),bins=self.n_bins)
			j = si.estimate_elbow_position(explained_variance_ratio=his*100/his.sum(),max_points=len(his))-1
			self.threshold = bins[j]
		else:
			self.threshold=threshold


		mask = fomd>self.threshold

		mask = expand_mask(mask)
		mask = expand_mask(mask)#2px margin

		kill_channels_from_mask(mask,self.core_loss_despiked)


		mask = self.core_loss_despiked.data<0
		if mask.sum()>15000:# to many spikes, something is off
			self.core_loss_despiked.data[mask]=0
		else:
			kill_channels_from_mask(mask,self.core_loss_despiked)
			mask = self.core_loss_despiked.data<0
			self.core_loss_despiked.data[mask]=0


		mask = self.core_loss_despiked.data<0
		self.core_loss_despiked.data[mask]=0
		self.despiked = True

		if self.long_despike>1:
			count_lim= self.long_despike
			self.long_despike = 0
			long_despike(self,count_lim=count_lim)

	def create_denoised(self):
		if self.despiked:
			self.core_loss_despiked.decomposition(True)
			self.denoised = self.core_loss_despiked.get_decomposition_model(self.core_loss_despiked.estimate_elbow_position())

		else:
			self.core_loss.decomposition()
			self.denoised = self.core_loss.decomposition(self.core_loss.estimate_elbow_position())

		self.denoise_applied=True

	def create_pyeels_fit(self,use="denoised"):
		if use =="denoised":
			data = self.denoised.data
		elif use =="despiked":
			data = self.core_loss_despiked.data
		elif use == "raw":
			data = self.core_loss.data
		else:
			print(" Can only use \"denoised\", \"despiked\", \"raw\"")
			raise AttributeError
		

		hl = em.MultiSpectrum.from_numpy(data,self.core_loss_despiked.axes_manager[-1].axis)
		ll = em.MultiSpectrum.from_numpy(self.low_loss.data,self.low_loss.axes_manager[-1].axis)
		
		self.hl = hl
		self.ll = ll

		if not hasattr(self,"comp_elements"):

			self.comp_elements = []
			A = 1
			#we have to use edges separately instead of combined
			for elem, edge in zip(self.elements, self.edges):
				comp = KohlLossEdgeCombined(hl.get_spectrumshape(), 1, self.E0, self.alpha,self.beta, elem, edge,fast=True)
				for x in comp.xsectionlist:
					x.cross_section=x.calculate_cross_section()
				#self.comp_elements.extend(comp.xsectionlist)
				self.comp_elements.append(comp)

		self.llcomp  = MscatterFFT(ll.get_spectrumshape(), ll,True,self.padding_mode_data,self.padding_mode_llspectrum)
	
		if not hasattr(self,"n_background"):
			print( "set self.n_background for pyeels Linear BG")

		bg = LinearBG(specshape=hl.get_spectrumshape(), rlist=np.linspace(1,self.n_background,self.n_background))

		self.comp_fine = []
		#for comp in comp_elements[0]:
		if not hasattr(self,"edges_with_fine_structure"):
			print("Set edges_with_fine_structure attribute")
			return

		
		for f in self.edges_with_fine_structure:
			element,edge=f
			corresponding =[i for i in self.comp_elements if (i.element==element) and i.edge==edge][0]

			if f in self.edges_with_fine_structure_parameters:
				pre_e =self.edges_with_fine_structure_parameters[f]["pre_e"]
				ewidth =self.edges_with_fine_structure_parameters[f]["ewidth"]
				degree =self.edges_with_fine_structure_parameters[f]["degree"]
				interpolationtype =self.edges_with_fine_structure_parameters[f]["interpolationtype"]
			else:
				pre_e = self.pre_e
				ewidth = self.ewidth
				degree = self.degree
				interpolationtype = self.interpolationtype

			cf = GDOSLin.gdoslin_from_edge(hl.get_spectrumshape(),
			corresponding, pre_e=pre_e, ewidth=ewidth, degree=degree,interpolationtype=interpolationtype)
			cf.element=element
			cf.edge=edge
			#cf.edge = what to do when multiple edges of the same element in spectrum? is this a problem?
			self.comp_fine.append(cf)

		component_list = [bg]+self.comp_elements+self.comp_fine

		if hasattr(self,"extra_fine_structures"):
			extras = []
			for d in self.extra_fine_structures:
				start = d["start"]
				ewidth = d["ewidth"]
				degree = d["degree"]
				extra_comp = GDOSLin(self.hl.get_spectrumshape(),start,ewidth,degree,interpolationtype=self.interpolationtype)
				extra_comp.element = d["element"]
				extra_comp.edge = d["edge"]
				extras.append(extra_comp)
			component_list += extras

		component_list+=[self.llcomp]
		mod = em.Model(hl.get_spectrumshape(), components=component_list)

		self.fit = LinearFitter(hl, mod, method=self.fit_method,use_weights=True)



	def average_spectrum_excluding_vacuum(self,signal,mask=None):
		if mask is None:
			mask = signal.data.sum(-1)>skimage.filters.threshold_otsu(signal.data.sum(-1))
		out = signal.inav[0,0]
		out.data = signal.data[mask].mean(0)
		self.mask_vacuum=mask
		self.average_spectrum_excluding_vacuum_signal=out
		return out 


	def remove_plural_scattering_via_model_fitting(self,element,edges):
		if not hasattr(self,"fit"):
			self.create_pyeels_fit()

		print("have you run self.fit.multi_fit()?")
		self.fit.original_llcomp = self.fit.model.components[-1]

		model = [i for i in self.fit.model.components if (check_element(i)==element and check_edge(i) in edges)]
		"""tmp = self.fit.model_to_multispectrum_with_comps(model).multidata
								self.fitted_edge = self.core_loss.deepcopy()
								self.fitted_edge.data = tmp"""


		

		if not hasattr(self,"llcomp_zlp"):
			zlp = self.low_loss.data.copy()
			zlp[:,:,np.invert(self.low_loss.get_zero_loss_peak_mask())]=0
			ll_zlp = em.MultiSpectrum.from_numpy(zlp,self.low_loss.axes_manager[-1].axis)
			llcomp_zlp = MscatterFFT(ll_zlp.get_spectrumshape(), ll_zlp,True,self.padding_mode_data,self.padding_mode_llspectrum)
			self.llcomp_zlp = llcomp_zlp
			self.fit.processing_llcomps = {"llcomp":self.llcomp,"llcomp_zlp":self.llcomp_zlp}

		self.fit.model.components[-1]=self.llcomp_zlp

		for c in self.fit.model.components:
			if not c in model:
				c.supressed=True
		tmp1 = self.fit.model_to_multispectrum().multidata
		for c in self.fit.model.components:
			c.supressed=False

		deconvolved_edge = self.core_loss.deepcopy()
		deconvolved_edge.data = tmp1
		deconvolved_edge.process_save_label=element+"_"+"_".join(edges)
		self.deconvolved_edge.append(deconvolved_edge)
		self.deconvolution_applied = True
		self.fit.model.components[-1]=self.llcomp
			
		return deconvolved_edge

def fom(data,n=7):
	fom = abs(np.diff(data)/moving_average(data,n)[...,:-1])
	return fom.shape[-1]*fom/fom.sum(-1)[:,:,np.newaxis]

def long_despike(process,dif_lim=0.5,count_lim=10):
	#Assumes initial despike has already been performed
	t0=process.threshold
	dif =np.inf
	count=0
	while dif>dif_lim and count_lim>count:
		count+=1
		process.despike(process.core_loss_despiked)
		dif=t0-process.threshold
		t0=process.threshold
		print("# Despikes : {}".format(count))

def check_element(comp):
	if hasattr(comp,"element"):
		return comp.element
	else:
		return None

def check_edge(comp):
	if hasattr(comp,"edge"):
		return comp.edge
	else:
		return None

def kill_channel(signal,coord):
	x,y,e=coord
	if e==0:
		signal.data[x,y,e]=signal.data[x,y,e+1]
	elif e==signal.data.shape[-1]-1:
		signal.data[x,y,e]=signal.data[x,y,e-1]
	else:
		signal.data[x,y,e]=(signal.data[x,y,e-1]+signal.data[x,y,e+1])/2

def kill_channel_range(signal,coord_range):
	x,y,er=coord_range
	ei,ef=er
	if ei==ef:
		kill_channel(signal,[x,y,ei])

	else:
		#check ranges:
		if ei-1<0:
			e_intp_i=ei
			I_intp_i = signal.data[x,y,ef+1]
		else:
			e_intp_i=ei-1
			I_intp_i = signal.data[x,y,ei-1]
			
		if ef+1>signal.data.shape[-1]-1:
			e_intp_f = ei
			I_intp_f = signal.data[x,y,ei-1]
		else:
			e_intp_f=ef+1
			I_intp_f = signal.data[x,y,ef+1]

		x_intp = np.linspace(e_intp_i,e_intp_f,e_intp_f-e_intp_i+1)
		I_intp = I_intp_i+((I_intp_f-I_intp_i)/(e_intp_f-e_intp_i))*(x_intp-x_intp[0])

		signal.data[x,y,e_intp_i:e_intp_f+1]=I_intp
	return

def kill_channels_from_mask(mask,signal):
	coords = np.transpose(mask.nonzero())
	coords = list([list (i) for i in coords])
	coords_set = set(tuple(c) for c in coords)
	coords2 = []
	coords2_set = set(coords2)
	print("Calulating spikes ranges...")
	for c in tqdm(coords):
		x,y,e=c

		if ((x,y,e+1) in coords_set) and ((x,y,e-1) in coords_set):
			pass
		elif (x,y,e+1) in coords_set:
			ei=e
			ef=e
			while (x,y,ef+1) in coords_set:
				ef+=1
			if (x,y,(ei,ef)) in coords2_set:
				pass
			else:
				coords2_set.add((x,y,(ei,ef)))
				
		elif (x,y,e-1) in coords_set:
			ei=e
			ef=e
			while (x,y,ei-1) in coords_set:
				ei-=1
			if (x,y,(ei,ef)) in coords2_set:
				pass
			else:
				coords2_set.add((x,y,(ei,ef)))
		else:
			coords2.append((x,y,(e,e)))

	print("Removing spikes...")
	for c2 in tqdm(list(coords2_set)):
		#print(c2)
		kill_channel_range(signal,c2)
	return


def expand_mask(mask):
	x,y,e=mask.shape
	mask1 = np.zeros((x,y,e+1))
	mask1[:,:,:-1]+=mask
	mask1[:,:,1:]+=mask
	return mask1>0


def load_results(folder,target_file,print_name=False,processing = None):
	"Assumes structure  /folder/tag1/tag2/.../tagn/n/targe_file"
	results={}
	extension = target_file.split(".")[-1]

	if extension in ["hspy","dm4","dm3"]:
		load=hs.load

	elif extension =="pkl":
		load = pkl_load
	elif extension == "npy":
		load = np.load

	else:
		print("extension not understood")
		return 

	for root, dirs, files in os.walk(folder):
		if target_file in files:
			tags = root.replace(folder,"").split(os.path.sep)[1:]

			d = results
			for t in tags:
				if t in d.keys():
					d = d[t]
				else:
					if t == tags[-1]:
						d[t]=[]
						d=d[t]
					else:
						d[t]={}
						d=d[t]

				if not isinstance(d,dict):
					if print_name:
						print(os.path.join(root,target_file))
					if not processing is None:
						d.append(processing(load(os.path.join(root,target_file))))
					else:
						d.append(load(os.path.join(root,target_file)))
	return results



def pkl_load(file):
	with open(file,"rb") as f:
		a = pkl.load(f)
	return a



def plot_results(results,**kwargs):
	"results is a dictionary with as many parameters as wanted i.e. results[material][beamcurrent][date]=[list of results]"
	flat = unravel(results)

	panels = len(flat.keys())

	r = int(np.sqrt(panels))
	c = int(np.sqrt(panels))


	while r*c!=panels:

		r=max(1,r-1)
		c=panels//r

	plt.clf()
	ii=0
	for k,l in flat.items():
		ii+=1
		ax = plt.subplot(r,c,ii)
		ax.set_title(k)
		for iii,s in enumerate(l):
			x = s.axes_manager[-1].axis
			y = norm(s.data)
			plt.plot(x,y+iii*0.5,**kwargs)

	


def unravel(d,tag=""):

	out={}
	for k,i in d.items():
		tag2 = tag+" "+k

		if isinstance(i,dict):
			out.update(unravel(i,tag2))

		else:
			out[tag2]=i


	return out

