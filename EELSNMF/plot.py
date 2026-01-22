from .imports import *
from .utils import *


class Plots:
    """ Mixin class for all plot functionalities of EELSNMF object."""
	def plot_factors(self):
		"""
		Plot EELSNMF factors spectral (mathematically the columns of G@W)
		"""
		plt.figure("Factors")
		plt.clf()
		for i in range(self.n_components):
			plt.plot(self.energy_axis,(self.G@self.W).T[i],label="Component {}".format(i))
		plt.legend()
		plt.tight_layout()


	def plot_loadings(self):
		"""
		Plot the aboundance maps of each component (mathematicall the rows of H)
		"""
		if not hasattr(self,"loadings"):
			self.calculate_loadings()

		self.plot_structure = find_2factors(self.n_components)
		plt.figure("Loadings")
		plt.clf()
		r,c= self.plot_structure
		for i in range(self.n_components):
			ax = plt.subplot(r,c,i+1)
			plt.imshow(self.loadings[i])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title("Loading {}".format(i))
		plt.tight_layout()


	def plot_edges(self,normalize=False):
		"""
		Plot the ELNES fit of each element and each component

		Parameters
		----------
		normalize : bool
			This flag rescales every ELNES from 0 to 1.
		     (Default value = False)

		"""
		plt.figure("Edges")
		plt.clf()
		r,c=find_2factors(len(self.edges))

		for ik,edge in enumerate(self.edges):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			for i in range(self.n_components):
				ii,ff=find_index(self.energy_axis,self.fine_structure_ranges[edge])
				if normalize:
					plt.plot(self.energy_axis[ii:ff],norm(self.get_edge_from_component(i,edge).data[ii:ff]))
				else:
					plt.plot(self.energy_axis[ii:ff],self.get_edge_from_component(i,edge).data[ii:ff])

		plt.tight_layout()


	def plot_chemical_maps(self,quantified=True,method=None):
		"""
		Plot the maps associated to the elemental concentration of each element
		Parameters
		----------
		quantified : bool
			If true, normalize each pixel by the sum of all chemical signals on that pixel (output is % of a chemical elment at each pixel)
		     (Default value = True)
		method : function or str
			Perform the quantification with this method, defaults are on EELSNMF.Analysis.default_q_methods
		     (Default value = None)

		"""
		
		qmaps=self.get_chemical_maps(quantified,method)
		plt.figure("Quantified Chemical Maps")
		plt.clf()
		r,c=find_2factors(len(self.edges))
		for ik,edge in enumerate(self.edges):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			plt.imshow(qmaps[edge])
			plt.colorbar()
		plt.tight_layout()

	def plot_average_model(self):
		"""Plot mean spectral signal and mean model """
		plt.figure("Model")
		plt.clf()
		plt.plot(self.energy_axis,self.X.mean(1),label="Data")
		plt.plot(self.energy_axis,self.get_model().mean(1),label="Model")
		plt.legend()
		plt.tight_layout()

	def plot_energy_ranges(self):
		"""
		Plot the energy ranges for each ELNES defined in self.fine_structure_ranges"""
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
		plt.tight_layout()
	









