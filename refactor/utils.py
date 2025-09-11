import numpy as np

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
