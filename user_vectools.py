import numpy as n
from scipy import signal
import pickle

def sort_vecs(data,vecs):
	"""Tri de matrice à plusieurs dimensions (3max)
	en fonction des vecteurs d'axes vecs"""
	data = n.array(data)
	for i in range(len(data)):
		if i < len(vecs) and vecs[i] is not None:
			xsort = vecs[i].argsort()
			vecs[i]=vecs[i][xsort]
			if i == 0: data = data[xsort]
			if i == 1: data = data[:,xsort]
			if i == 2: data = data[:,:,xsort]
	return data,vecs

def crop_vecs(data,vecs,crop):
	"""Découpage des données en fonction des vecteurs d'axe"""
	data = n.array(data)
	N = len(n.shape(data))
	for i in range(N):
		if crop[2*i] == None: crop[2*i] = n.min(vecs[i])
		if crop[2*i+1] == None: crop[2*i+1] = n.max(vecs[i])
		xsort = n.where((vecs[i] >= crop[2*i]) & (vecs[i] <= crop[2*i+1]))[0]
		vecs[i]=vecs[i][xsort]
		if i == 0: data = data[xsort]
		if i == 1: data = data[:,xsort]
		if i == 2: data = data[:,:,xsort]
	return data,vecs


def set_to_lim(data,min_lim = None,max_lim=None):
	"""Mise des données à une valeurs limite lorsque celles ci ne sont plus dans la gamme définie"""
	data = n.array(data)
	if min_lim is not None: data[n.where(data<=min_lim[0])] = min_lim[1]
	if max_lim is not None: data[n.where(data>=max_lim[0])] = max_lim[1]
	return data

def resamp_vecs(data,vecs,N,sort_data=True):
	"""Sampling des données"""
	for i in range(len(N)):
		if N[i] is not None:
			data = signal.resample(data,N[i],axis=i)
			if vecs[i] is not None: vecs[i] = signal.resample(vecs[i],N[i])
	if sort_data: data,vecs = sort_vecs(data,vecs)
	return data,vecs

def save_obj(obj, name ):
	"""Sauvegarde d'une instance"""
	with open( name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
	"""Chargement d'une instance"""
	with open( name + '.pkl', 'rb') as f:
		return pickle.load(f)

def put_mat_in_vec(data):
	"""Linéarisation d'une matrice"""
	return n.reshape(data,int(n.shape(data)[0]*n.shape(data)[1]))

def normL3(res):
	"""Norme L3"""
	return res/n.max(n.abs(res))

def highlight_freq(data,fvec,axis=-1,flow=None,fhigh=None,coef=0.9):
	"""Mise en evidence d'une plage de fréquence"""
	if axis == -1:
		if fhigh is not None:
			ysort = n.where((fvec > fhigh))[0]
			for i in range(n.size(data,0)):
				data[i][ysort] = data[i][ysort]*coef
		if flow is not None:
			ysort = n.where((fvec < flow))[0]
			for i in range(n.size(data,0)):
				data[i][ysort] = data[i][ysort]*coef
	elif axis == -2:
		if flow is not None:
			ysort = n.where((fvec > fhigh))[0]
			data[ysort] = data[ysort]*coef
		if fhigh is not None:
			ysort = n.where((fvec < flow))[0]
			data[ysort] = data[ysort]*coef
	return data

