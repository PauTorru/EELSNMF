import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from .utils import *


######################### many needs to modify everything


class Plots:
	def plot_factors(self):
		plt.figure("Factors")
		plt.clf()
		for i in range(self.n_components):
			plt.plot(self.energy_axis,(self.G@self.W).T[i],label="Component {}".format(i))
		plt.legend()
		plt.tight_layout()


	def plot_loadings(self):
		if not hasattr(self,"loadings"):
			self.calculate_loadings()
		if not hasattr(self,"plot_structure"):
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
		plt.figure("Edges")
		plt.clf()
		r,c=find_2factors(len(self.edges))

		for ik,edge in enumerate(self.edges):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			for i in range(self.n_components):
				ii,ff=self.find_index(self.fine_structure_ranges[edge])
				if normalize:
					plt.plot(self.energy_axis[ii:ff],norm(self.get_edge_from_component(i,edge).data[ii:ff]))
				else:
					plt.plot(self.energy_axis[ii:ff],self.get_edge_from_component(i,edge).data[ii:ff])

		plt.tight_layout()


	def plot_chemical_maps(self,quantified=True):
		
		qmaps=self.get_chemical_maps(quantified)
		plt.figure("Quantified Chemical Maps")
		plt.clf()
		r,c=find_2factors(len(self.edges))
		for ik,edge in enumerate(self.edges):
			ax = plt.subplot(r,c,ik+1)
			ax.set_title(edge)
			plt.imshow(qmaps[edge])
			plt.colorbar()

	def plot_average_model(self):
		plt.figure("Model")
		plt.clf()
		plt.plot(self.energy_axis,self.X.mean(1),label="Data")
		plt.plot(self.energy_axis,self.get_model().mean(1),label="Model")
		plt.legend()

	def plot_energy_ranges(self):
		plt.figure("Edges energy ranges")
		plt.clf()
		plt.plot(self.energy_axis,norm(self.X.mean(1)))
		ax=plt.gca()
		colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
		c=-1
		for k,v in self.model.fine_structure_ranges.items():
			c+=1
			ii,ff = v
			rect = mpl.patches.Rectangle((ii, 0.), ff-ii, 1,
                         alpha=0.5,label=k,color=colors[c])
			ax.add_patch(rect)
		ax.set_xlim(self.energy_axis.min(),self.energy_axis.max())
		ax.set_ylim(0,1)
		ax.set_yticks([])
		plt.legend()
	









