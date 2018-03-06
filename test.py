class personne():
	nom = None
	prenom = None
	def __init__(self,**kwargs):
		self.nom = kwargs.get("nom",None)
		self.prenom = kwargs.get("prenom",None)

class homme(personne):
	age=0
	def __init__(self,**kwargs):
		kwargs["nom"]="tre"
		super().__init__(**kwargs)
		self.age = kwargs.get("age",0)
		# print(kwargs,op)


# a = dict()
# a["nom"] = "frank"
b = homme(nom="franck",age=15)

print(b.nom)