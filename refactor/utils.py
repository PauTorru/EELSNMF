import numpy as np
import scipy as sc

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
