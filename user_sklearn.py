import numpy as n
from matplotlib import pyplot as plt
from scipy import signal as sig
import time
import sys
from sklearn import preprocessing as proc,svm
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV 
from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from multiprocessing import cpu_count
import pickle

class trainData():
	"""Classe gérant l'entrainement et l'optimisation de modele scikit-learn en regression

	Classe permettrant d'entrainer un modele, de réduire la dimension des données d'entrée,
	la sauvegarde et le chargement.

	"""
	input_data = None
	target = None
	err = None
	fitted_model = None
	fitted_red = None
	params = None

	def __init__(self,data_file,target_file,err=0.01):
		"""Constructeur"""
		self.input_data = data_file
		self.target = target_file
		self.err = err

	def try_model(self,model="Lasso",model_params=None,dim_reduction=None,with_plot=False,Niter=100):
		""" Fonction permettrant d'entrainer un modele et de réduire la dimension

		Modeles disponibles : Lasso, ElasticNet, Ridge, SVR, RF
		"""
		# PCA
		if dim_reduction is not None:
			if dim_reduction[0] == "PCA":
				pca = PCA(n_components=dim_reduction[1])
				self.fitted_red = pca.fit(self.input_data)
				self.input_data = self.fitted_red.transform(self.input_data)

		# Models
		## Find params
		if model is not None:
			if model_params is None:
				if model == "Lasso":
					params = {"alpha":{"start":0,"end":1000}}
					clf = self.user_grid_search(Lasso(),params,self.err)
				if model == "ElasticNet":
					params = {"alpha":{"start":0.00001,"end":1000},"l1_ratio":{"start":0,"end":1}}
					clf = self.user_grid_search(ElasticNet(),params,self.err)
				if model == "Ridge":
					params = {"alpha":{"start":0,"end":1000}}
					clf = self.user_grid_search(Ridge(),params,self.err)
				if model == "SVR" or model == "SVM":
					params = {"epsilon":{"start":0,"end":1},"C":{"start":0.1,"end":1000}}
					clf = self.user_grid_search(svm.SVR(kernel="linear"),params,self.err)
				if model == "RF":
					params = {"n_estimators":n.arange(2,100,1)}
					clf = self.user_grid_search(RandomForestRegressor(),params,self.err)
				
				print("Best params for",model)
				for ii,jj in clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()].items():
					print(ii,": {0:9.9f}".format(jj))
				self.params = clf.cv_results_["params"][clf.cv_results_["mean_test_score"].argmax()]
			else:
				self.params = model_params

			## Test model
			ares = list()		
			for i in range(Niter):	
				fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(self.input_data,self.target,test_size=0.33)
				if model == "Lasso": clf = Lasso(**self.params)
				if model == "ElasticNet": clf = ElasticNet(**self.params)
				if model == "Ridge": clf = Ridge(**self.params)
				if model == "SVR" or model == "SVM": clf = svm.SVR(kernel="linear",**self.params)
				if model == "RF": clf = RandomForestRegressor(**self.params)
				self.fitted_model = clf.fit(fmxtrain, fmytrain)	
				lres = list()
				for j in range(Niter):
					fmxtrain, fmxtest,fmytrain,fmytest = train_test_split(self.input_data,self.target,test_size=0.33)
					lres.append(r2_score(fmytest, self.fitted_model.predict(fmxtest)))
				ares.append(n.mean(lres))
				print(ares[-1])

			s = 1-n.sum(n.abs(self.fitted_model.predict(self.input_data)-self.target)/self.target)/len(self.target)
			ss = n.std(1-n.abs(self.fitted_model.predict(self.input_data)-self.target)/self.target)
			print(1-n.abs(self.fitted_model.predict(self.input_data)-self.target)/self.target)
			print("Moyenne des scores r2 =",n.mean(ares),"+-",n.std(ares))
			print("Moyenne des écarts relatifs =",s,"+-",ss)
			## Figures
			if with_plot:
				plt.figure()
				plt.subplot(211)
				plt.plot(self.target,n.abs(self.fitted_model.predict(self.input_data)-self.target)/self.target*100,"x")
				plt.xlabel("Flowrates (m3/h)")
				plt.ylabel("Relative uncertainty%")
				plt.title(model + " model" + " -> "+"{:.2%}".format(s))

				plt.grid()
				plt.subplot(212)
				# plt.title("Model : "+model)
				plt.plot(self.target,self.fitted_model.predict(self.input_data),"x")
				plt.xlabel("Desired Flowrates (m3/h)")
				plt.ylabel("Predicted Flowrates (m3/h)")
				plt.grid()
				plt.show()

	def user_grid_search(self,skmod,params,max_err=0.01,n_jobs=1):
		"""Fonction d'optimisation des paramètres des modèles"""
		parameters = dict()
		err = 10
		for i,j in params.items():
			if isinstance(j,dict):
				parameters[i] = n.arange(j["start"],j["end"]+n.abs(j["end"]-j["start"])/10,n.abs(j["end"]-j["start"])/10)
			else:
				if isinstance(j,list) or isinstance(j,n.ndarray):
					parameters[i] = j
				else:
					parameters[i] = [j] 
		print(parameters)
		while err > max_err :
			clf = GridSearchCV(skmod,parameters,n_jobs=n_jobs)
			clf.fit(self.input_data,self.target)
			N1 = clf.cv_results_["mean_test_score"].argmax()
			err = n.sqrt(n.var(clf.cv_results_["mean_test_score"]))
			print(clf.cv_results_["mean_test_score"])
			print("Max mean score : ",n.max(clf.cv_results_["mean_test_score"]))
			print("Err : ",err)
			for i,j in clf.cv_results_["params"][N1].items():
				if isinstance(params[i],list) or isinstance(params[i],n.ndarray): 
					err = 0
					break
				if not isinstance(j,str):
					if len(parameters[i]) > 1:
						dx = n.abs(parameters[i][1]-parameters[i][0])
						if j-dx < parameters[i][0]: par1 = j
						else: par1 = j-dx
						if j+dx > parameters[i][-1]: par2 = j
						else: par2 = j+dx
						parameters[i]= n.arange(par1,par2,n.abs(par1-par2)/10)
					else:

						parameters[i] = [j]
				else:
					parameters[i] = [j]
				print(i,parameters[i])
				print(i+" max :",j)
		return clf

	def user_score(self,target,pred):
		"""Calcul de la performance de prédiction"""
		S = 0
		N = len(target)
		for i in range(N):
			S += n.abs(target[i]-pred[i])/target[i]
		return S/N

	def save_model(self,fpath,model="pred"):
		"""Sauvegarde du modèle généré"""
		try:
			f = open(fpath, 'wb')
		except:
			print("Chemin invalide :",fpath)
			return True
		if model == "pred" and self.fitted_model is not None: pickle.dump(self.fitted_model, f)
		if model == "red" and self.fitted_red is not None: pickle.dump(self.fitted_red, f)
		f.close()
		return False	

	def load_model(self,fpath,model="pred"):
		"""Chargement d'un modèle"""
		try:
			f = open(fpath, 'rb')
		except:
			print("Chemin invalide :",fpath)
			return True

		if model == "pred":	self.fitted_model = pickle.load(f)
		if model == "red":	self.fitted_red = pickle.load(f)
		return False