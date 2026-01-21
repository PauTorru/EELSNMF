from .imports import *
from .modelG import ModelG
from .decompositions import Decomposition
from .plot import Plots
from .utils import *
from .analysis import Analysis




def load(fname):
	with open(fname,"rb") as f:
		return pkl.load(f)

class EELSNMF(ModelG,Decomposition,Plots,Analysis):

	def __getstate__(self): #to avoid objects during pickling
		state = self.__dict__.copy()
		#state = self.__dict__.copy()
		for i in self._hspy:
			if i in state.keys():
				del state[i]
		if "xp" in state.keys():
			del state["xp"]
		return state

	def __setstate__(self,state):
		self.__dict__.update(state)



	def __init__(self,core_loss, 
								E0 = None,
								alpha = None,
								beta = None,
								):

		"""
		Class to perform EELSNMF analysis of EELS core-loss data.


		Parameters
		----------

		core_loss: hs.signals.Signal1D
			Spectrum image to which the EELSNMF decomposition will be applied.


		E0, alpha, beta: float
			Acceleration voltage, convergence angle and collection angle of the acquistion.
			If None, it is read from image metadata.
			Units are respectively V, rad, and rad. (not kV, mrad)

		"""
		super().__init__()

		self.cl = core_loss

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


		self.energy_size = self.cl.data.shape[-1]
		self.energy_axis = self.cl.axes_manager[-1].axis

		self._eaxis_parameters = {
		"offset":self.cl.axes_manager[-1].offset,
		"scale":self.cl.axes_manager[-1].scale,
		"name":self.cl.axes_manager[-1].name,
		"units":self.cl.axes_manager[-1].units,
		"axis": self.energy_axis
		}





		self.spatial_shape = self.cl.data.shape[:-1]

		assert self.cl.data.min()>=0

		self.dtype =np.float64
		self.X = self.cl.data.reshape((-1,self.energy_size)).T.astype(self.dtype)

		self._m = ["G","X","W","H"] 
		self._hspy = ["cl","ll"] #will be skipped when pickling

		self.analysis_description = {}



	@property
	def G(self):
		return self.model.G

	@G.setter
	def G(self, value):
		self.model.G = value
	
	def change_dtype(self,dtype):

		for m in self._m:
			if hasattr(self,m):
				setattr(self,m,getattr(self,m).astype(dtype))
		return

	def save(self, fname, path=None, save_hspy_objects = False,overwrite=False):
		"""
		Save the EELSNMF object. This method uses pickle.
		EELSNMF objects contain some hyperspy classes (self.cl). Usually these are not necessary once the analysis has been performed,
		since all the parameters needed from them are stored as separate attributes. Therefore saving them is not necessary and inconvenient since they can't be pickled.

		Paramters:
		----------

		fname: str
			name of the file to be saved

		path: str
			path where the file will be saved

		save_hspy_objects: bool
			Save cl,ll objects separaterly

		overwrite: bool
			Check if file exists and decide to overwrite it or not
		"""
		if fname[-4:]==".pkl":
			fname=fname[:-4]

		if path is None:
			path = os.getcwd()

		if os.path.exists(os.path.join(path,fname+".pkl")) and not overwrite:
			print("File already exists")
			return

		for ss in self._hspy:
			if hasattr(self,ss):
				s = getattr(self,ss)
				if save_hspy_objects:
					s.save(os.path.join(path,"_".join([fname,ss,".hspy"])))

		with open(os.path.join(path,fname+".pkl"),"wb") as f:
			pkl.dump(self,f)












