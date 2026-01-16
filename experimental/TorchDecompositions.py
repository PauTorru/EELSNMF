from ..imports import *
import torch
import torch.nn as nn
import torch.optim as optim


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
