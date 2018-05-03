import sys
import os
import math
from PyQt5.QtWidgets import QApplication, QScrollArea, QMainWindow,QSizePolicy, QPushButton,QComboBox,QSlider, QStyle, QMessageBox,QLineEdit,QWidget,QLabel,QDesktopWidget,QAction,QFrame,QBoxLayout,QVBoxLayout,QHBoxLayout,QGridLayout,QGraphicsColorizeEffect,QGroupBox,QRadioButton,QCheckBox
from PyQt5.QtGui import QIcon, QPixmap,QPainter,QPalette,QColor,QBrush,QPen
from PyQt5.QtCore import Qt,QObject,QSize,QPropertyAnimation,QRect
usty = None
utxt = None
ulang = None


def txtBox(title,corpus,qtype="info",parent=None):
	if title in utxt: title = utxt[title][ulang]
	if corpus in utxt: corpus = utxt[corpus][ulang]
	if qtype == "about":
		QMessageBox.about(parent,title,corpus)
	if qtype == "critical":
		QMessageBox.critical(parent,title,corpus,QMessageBox.Ok)
	if qtype == "info":
		QMessageBox.information(parent,title,corpus,QMessageBox.Ok)				
	if qtype == "question":
		r = QMessageBox.question(parent,title,corpus,QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
		if r == QMessageBox.Yes:
			return 1
		if r == QMessageBox.No:
			return 0
		if r == QMessageBox.Cancel:
			return -1
	if qtype == "warning":
		QMessageBox.warning(parent,title,corpus,QMessageBox.Ok)

def get_text(name_id):
	if name_id in utxt and ulang in utxt[name_id]:
		return utxt[name_id][ulang]
	else:
		return name_id


class UQobject(QObject):
	def __init__(self,*args,**kwargs):
		super().__init__()
		kwargs = self._args(*args,**kwargs)
		name_id = kwargs.get("name_id",None)
		parent = kwargs.get("parent",None)
		if name_id is not None: self.setObjectName(name_id)
		if parent is not None: self.setParent(parent)

	def _args(*args,**kwargs):
		if "name_id" not in kwargs: 
			if len(args) > 1: kwargs["name_id"] = args[1]
		if "name_id" in kwargs and kwargs["name_id"] in usty:  
			kwargs.update(usty[kwargs["name_id"]])
		if "name_id" in kwargs and "title" not in kwargs and kwargs["name_id"] in utxt:
			kwargs["title"] = kwargs["name_id"]
		if "title" in kwargs:
			if kwargs["title"] in utxt:
				if ulang in utxt[kwargs["title"]]: 
					kwargs["title"] = utxt[kwargs["title"]][ulang]
		return kwargs

	def get_child_by_name(self,name_id):
		for i in self.children():
			if str(i.objectName()) == str(name_id):
				return i
		return None			



class UQapplication(QApplication,UQobject):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)



class UQapp(QMainWindow,UQobject):
	"""
		Classe Fenetre principale
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		title = kwargs.get("title",None)
		if title is not None: self.setWindowTitle(title)
		self.want_to_close = None


	def status_txt(self,status_txt):
		"""
			Message dans la barre de status
		"""
		self.statusBar().showMessage(status_txt)

	def __setattr__(self,nom,val):
		super(UQapp,self).__setattr__(nom,val)
		if nom == 'title':
			self.setWindowTitle(self.p_title)
		if nom == 'left':
			self.move(val,self.top)
		if nom == 'top':
			self.move(self.left,val)
		if nom == 'width':
			self.resize(val,self.height)
		if nom == 'height':
			self.resize(self.width,val)
		if nom == 'style':
			if os.path.isfile(val + "/" + "temp.css"):
				os.remove(val + "/" + "temp.css")
			with open(val + "/" + "temp.css", 'w') as outfile:
			    l = os.listdir(val)
			    for fname in l:
			        with open(val + "/" + fname) as infile:
			            for line in infile:
			                outfile.write(line)
			outfile.close()
			with open(val+"/"+"temp.css") as fh:
			    self.setStyleSheet(fh.read())

	def center(self,screen_num=0):
		"""
			Centre la fenetre dans le l'écran screen_num
		"""
		qr = self.frameGeometry()
		cp = QDesktopWidget().screenGeometry(screen_num).center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def read_css(path):
		with open(path) as fh:
		    app.setStyleSheet(fh.read())

	def keyPressEvent(self, e):   
		if e.key() == Qt.Key_F11:
			self.showMaximized()
		elif e.key() == Qt.Key_Escape:
			self.showNormal()        

	def closeEvent(self, evnt):
		if self.want_to_close is None:
			super().closeEvent(evnt)
		else:
			evnt.ignore()
			self.want_to_close()

class UQwidget(QWidget,UQobject):
	offset = None
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		name_id = kwargs.get("name_id",None)
		title = kwargs.get("title",None)
		style = kwargs.get("style",None)
		policy = QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
		if title is not None:self.setText(title)
		else: policy = QSizePolicy(0,0)
		if style is not None: self.setProperty("class",style)
		self.setSizePolicy(policy)

class UQdragwidget(UQwidget):
	offset = None
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.setSizePolicy(0,QSizePolicy.Fixed)

	def mousePressEvent(self, event):
		self.offset = event.pos()

	def mouseMoveEvent(self, event):
		if event.buttons() == Qt.LeftButton and self.offset is not None:
			self.move(self.mapToParent(event.pos()-self.offset))

class UQfixwidget(UQwidget):
	def __init__(self,*args,**kwargs):
		# super().__init__(name_id,parent,title,style)
		super().__init__(*args,**kwargs)
		self.setSizePolicy(0,QSizePolicy.Fixed)

class UQtxt(QLabel,UQwidget):
	"""
		Classe définissant un champs text non-editable
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.show()


class UQdragline(UQtxt):
	offset = None
	redim = None
	iniwidth = None
	iniheight = None
	xini = None
	yini = None
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.redim = kwargs.get("redim",None)
		self.posi = kwargs.get("posi",None)
		self.size = kwargs.get("size",None)
		self.iniwidth = self.width()
		self.iniheight = self.height()
		if self.size is not None:
			if self.redim == "w": self.resize(self.size,self.iniheight)
			if self.redim == "h": self.resize(self.iniwidth,self.size)
		if self.posi is not None: self.move(self.posi[0],self.posi[1])

	def mousePressEvent(self, event):
		self.offset = event.pos()

	def mouseMoveEvent(self, event):
		if event.buttons() == Qt.LeftButton and self.offset is not None:
			self.move(self.mapToParent(event.pos()-self.offset))
		if event.buttons() == Qt.RightButton and self.offset is not None:
			x2 = int(event.pos().x())			
			y2 = int(event.pos().y())
			w = abs(x2) 
			h = abs(y2)
			if w == 0 : w = 10
			if h == 0 : h = 10
			if self.redim == "w": h = self.iniheight
			if self.redim == "h": w = self.iniwidth
			self.repaint()
			self.resize(w,h)

class UQline(UQtxt):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		redim = kwargs.get("redim",None)
		posi = kwargs.get("posi",None)
		size = kwargs.get("size",None)
		anim = kwargs.get("anim",True)
		w = 5
		if redim == "w": self.resize(size,self.height())
		if redim == "h": self.resize(self.width(),size)

		self.move(posi[0],posi[1])
		ef = QGraphicsColorizeEffect(self)
		self.setGraphicsEffect(ef)
		self.__animation = QPropertyAnimation(ef,b"color")
		self.__animation.setDuration(2000)
		self.__animation.setKeyValueAt(0,QColor(0, 255, 0))
		self.__animation.setKeyValueAt(0.5,QColor(255, 0, 0))
		self.__animation.setKeyValueAt(1,QColor(0, 255, 0))
		self.__animation.setLoopCount(-1)
		if anim: self.__animation.start()
		self.show()

	def load_anim(self):
		self.__animation.start()

	def stop_anim(self):
		self.__animation.stop()

class UQtxtedit(QLineEdit,UQwidget):
	"""
		Classe définissant un champs text editable
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.show()

class UQslider(QSlider,UQwidget):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		vmin = kwargs.get("vmin",None)
		vmax = kwargs.get("vmax",None)
		step = kwargs.get("step",None)
		direc = kwargs.get("step",Qt.Horizontal)
		connect2 = kwargs.get("connect2",None)
		if vmin is not None: self.setMinimum(vmin)
		if vmax is not None: self.setMaximum(vmax)
		if step is not None: self.setSingleStep(step)
		if connect2 is not None:
			if connect2[0] == "changed":
				self.valueChanged.connect(connect2[1])
		self.setOrientation(direc)

class UQbut(QPushButton,UQwidget):
	"""
		Classe définissant le widget bouton
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		title = kwargs.get("title",None)
		icon = kwargs.get("icon",None)
		tooltip = kwargs.get("tooltip",None)
		connect2 = kwargs.get("connect2",None)
		expand = kwargs.get("expand",None)
		if expand is False: self.setSizePolicy(0,0)

		if tooltip is not None:self.setToolTip(tooltip)
		elif title is not None: self.setToolTip(title)
		if icon is not None: self.p_icon = icon
		if connect2 is not None:
			if connect2[0] == "clicked":
				self.clicked.connect(connect2[1])	
		self.show()

	def __setattr__(self,nom,val):
		super().__setattr__(nom,val)
		if nom == 'p_icon' and val != None:
			if len(str(val).split(".")) == 1 :
				self.setIcon(self.style().standardIcon(val))
			else:
				self.setIcon(QIcon(QPixmap(val)))
			self.setIconSize(QSize(32,32))
	def enterEvent(self,event):
		self.setCursor(Qt.PointingHandCursor)

class UQcombo(QComboBox,UQwidget):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		items = kwargs.get("items",None)
		connect2 = kwargs.get("connect2",None)
		if items is not None: self.addItems(items)
		if connect2 is not None:
			if connect2[0] == "changed":
				self.currentIndexChanged.connect(connect2[1])
		self.setSizePolicy(QSizePolicy.Expanding,0)


class UQgroupbox(QGroupBox,UQwidget):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class UQradio(QRadioButton,UQwidget):
	def __init__(self,*args,**kwargs):
		kwargs = self._args(*args,**kwargs)
		super().__init__(*args,**kwargs)
		connect2 = kwargs.get("connect2",None)
		# style = kwargs.get("style",None)
		# title = kwargs.get("title",None)
		# if title is not None:self.setText(title)
		# if style is not None: self.setProperty("class",style)
		self.setAutoExclusive(True)
		if connect2 is not None:
			if connect2[0] == "clicked":
				self.clicked.connect(connect2[1])	
			if connect2[0] == "toggled":
				self.toggled.connect(connect2[1])

class UQcheckbox(QCheckBox,UQwidget):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		connect2 = kwargs.get("connect2",None)
		style = kwargs.get("style",None)
		self.setAutoExclusive(True)
		if connect2 is not None:
			if connect2[0] == "clicked":
				self.clicked.connect(connect2[1])	
			if connect2[0] == "toggled":
				self.toggled.connect(connect2[1])

class UQaction(QAction,UQobject):
	"""
		Classe de définition d'action
	"""
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		kwargs = self._args(*args,**kwargs)
		title = kwargs.get("title",None)
		icon = kwargs.get("icon",None)
		shortcut = kwargs.get("shortcut",None)
		if icon is not None: self.setIcon(QIcon(icon))
		if title is not None:
			self.setStatusTip(title)
			self.setText(title)
		if shortcut is not None: self.setShortcut(shortcut)

class UQframebox(QFrame,UQwidget):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		policy = QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
		self.setSizePolicy(policy)
		self.setSizePolicy(0, 0)
		self.show()

class UQscrollarea(QScrollArea,UQframebox):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class UQboxlayout(QBoxLayout,UQobject):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)
		self.setContentsMargins(0,0,0,0)
		self.setSpacing(0)

	def get_item_by_name(self,name_id):
		for i in range(self.count()):
			if self.itemAt(i).widget().objectName() == name_id:
				return self.itemAt(i)
		return None

	def get_widget_by_name(self,name_id):
		w = self.get_item_by_name(name_id)
		if w is not None:
			return w.widget()
		return None

	def get_widget_by_pos(self,i,j=None):
		if j is not None:
			itm = self.itemAtPosition(i,j)
		else:
			itm = self.itemAt(i)
		if itm is None:
			return None
		else:
			return itm.widget()

class UQvboxlayout(QVBoxLayout,UQboxlayout):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class UQhboxlayout(QHBoxLayout,UQboxlayout):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

class UQgridlayout(QGridLayout,UQboxlayout):
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)

	def get_widget_position(self,name_id):
		w = self.get_widget_by_name(name_id)
		if w is None:
			return None
		wid = self.indexOf(w)
		if wid == -1:
			return None
		row = col = None 
		l =  self.getItemPosition(wid)
		if l is None:
			return None
		return l


class Overlay(QWidget):
	status = True
	def __init__(self, parent = None):
		QWidget.__init__(self, parent)
		palette = QPalette(self.palette())
		palette.setColor(palette.Background, Qt.transparent)
		self.setPalette(palette)
		self.setGeometry(int(parent.width()/2+parent.pos().x()/2-50),int(parent.height()/2+parent.pos().y()/2-50),100,100)

	def paintEvent(self, event):      
		painter = QPainter()
		painter.begin(self)
		painter.setRenderHint(QPainter.Antialiasing)
		painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 127)))
		painter.setPen(QPen(Qt.NoPen))
       
		for i in range(6):
			if (self.counter / 5) % 6 == i:
				painter.setBrush(QBrush(QColor(127 + (self.counter % 5)*32, 127, 127)))
			else:
				painter.setBrush(QBrush(QColor(127, 127, 127)))
				painter.drawEllipse(
				self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
				self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
				20, 20)
		painter.end()
   
	def showEvent(self, event):      
		# self.timer = self.startTimer()
		self.counter = 0
 
	# def visu_update(self):
	# 	print("yes")
	# 	self.counter += 1
	# 	self.update()
		# if not self.status:
		# 	# self.killTimer(self.timer)
		# 	self.hide()
	
	# def stop():
	# 	self.status = False