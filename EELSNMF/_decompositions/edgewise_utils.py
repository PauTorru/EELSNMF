from ..imports import *
from ..utils import find_index


class EdgeWiseUtils:

	def _build_S(self):
			""" S_i are the subgroups on which the penalty will be applied """
			self._edge_indices = {}

			for edge in self.edges:
				mask = np.zeros_like(self.G[:,0],dtype="bool")
				#mask[self.model.xsection_idx[edge]]=True
				mask[self.model._edge_slices[edge]]=True
				self._edge_indices[edge] =np.where(mask)[0]# np.roll(np.where(mask)[0],-1) #moves xsection to the end for TV

			return

