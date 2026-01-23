from ..imports import *

#attr_list = ["GtX","GtG","X","W","G","H","W_init","W_fixed_bool","W_fixed_values","GW","X_over_GWH","GTsum1"]

class Cupy_Utils:

	def _np2cp(self):
		self.xp = cp
		for attr in self._m:
			value = getattr(self,attr,None)
			if value is not None:
				setattr(self,attr,cp.array(value))

		#self.GtX = cp.array(self.GtX)
		#self.GtG = cp.array(self.GtG)
		#self.X = cp.array(self.X) 
		#self.W = cp.array(self.W)
		#self.G = cp.array(self.G)
		#self.H = cp.array(self.H)

		#if hasattr(self,"W_init"):
		#	if not self.W_init is None:
		#		self.W_init=cp.array(self.W_init)
		#if hasattr(self,"W_fixed_bool"):
		#	if not self.W_fixed_bool is None:
		#		self.W_fixed_bool=cp.array(self.W_fixed_bool)
		#if hasattr(self,"W_fixed_values"):
		#	if not self.W_fixed_values is None:
		#		self.W_fixed_values=cp.array(self.W_fixed_values)


	def _cp2np(self):
		self.xp=np
		for attr in self._m:
			value = getattr(self,attr,None)
			if value is not None:
				setattr(self,attr,value.get())
		clear_gpu_cache()

		#self.X = self.X.get()
		#self.W = self.W.get()
		#self.G = self.G.get()
		#self.H = self.H.get()
		#self.GtG  = self.GtG.get()
		#self.GtX  = self.GtX.get()
		#if hasattr(self,"W_init"):
		#	if not self.W_init is None:
		#		self.W_init=self.W_init.get()
		#if hasattr(self,"W_fixed_bool"):
		#	if not self.W_fixed_bool is None:
		#		self.W_fixed_bool=self.W_fixed_bool.get()
		#if hasattr(self,"W_fixed_values"):
		#	if not self.W_fixed_values is None:
		#		self.W_fixed_values=self.W_fixed_values.get()

def clear_gpu_cache():

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
