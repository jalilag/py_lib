import sqlite3 as sql
import re
import numpy
class Usql:
	"""
	Classe permettant de réaliser des opérations sur une base SQLite3
	"""
	cursor = None
	def __init__(self,conn):
		"""Constructeur"""
		self.conn = conn
		self.conn.row_factory = self.dict_factory
		self.cursor = self.conn.cursor()
		self.cursor.execute("""SELECT name  FROM sqlite_master WHERE type = 'table'""")
		self.tbl_name = numpy.ravel(self.cursor.fetchall())

	def dict_factory(self,cursor, row):
		d = {}
		for idx, col in enumerate(cursor.description):
		    d[col[0]] = row[idx]
		return d

	def cols(self,table_name):
		"""Récupérer la liste des noms de colonne d'une table"""
		# self.cursor.execute("SELECT sql  FROM sqlite_master WHERE type = 'table' AND name = '" + table_name +"'")
		c = self.cursor.execute("SELECT *  FROM " + table_name)
		tables = list(map(lambda x: x[0], c.description))
		return tables

	def cols_info(self,table_name):
		return self.request("PRAGMA table_info("+table_name+")")

	def build_order(self,u_order,u_from,u_item="*",u_where=None,u_limit=None,u_join=None,u_orderby=None):
		s =u_order + " " + u_item + " FROM " + u_from
		if u_join is not None:
			s += " " + u_join 
		if u_where is not None:
			s += " WHERE " + u_where
		if u_orderby is not None:
			s += " ORDER BY " + u_orderby
		if u_limit is not None:
			s += " LIMIT " + u_limit
		return s

	def check_values(self,table_name,val_dict):
		cols_info = self.cols_info(table_name)
		for i in cols_info:
			if i["name"] in val_dict:
				val_dict[i["name"]] = str(val_dict[i["name"]]) 
				if (val_dict[i["name"]] is None or val_dict[i["name"]] == "") and i["notnull"]: return [0x0001,True,i["name"]]
				if val_dict[i["name"]] is None or val_dict[i["name"]] == "":
					val_dict[i["name"]] = "NULL"
					continue
				if i["type"] in ['TEXT','DATE']:
					val_dict[i["name"]] = "'" + val_dict[i["name"]] + "'"
				if i["type"] == 'BOOLEAN':
					val_dict[i["name"]] = ("0","1")[val_dict[i["name"]] == "True"]
				if i["type"] == "INTEGER":
					if not val_dict[i["name"]].isdigit(): return [0x0002,True,i["name"]]
				if i["type"] == "DOUBLE":
					try:
						float(val_dict[i["name"]])
					except:
						return [0x0003,True,i["name"]]
				# Custom data_types

		return [0,False,None]

	def remove_prefix(self,word):
		re

	def request(self,string,just_values = False):
		self.cursor.execute(string)
		s = self.cursor.fetchall()
		if just_values:
			ss = list()
			for i in s:
				ss.append(list(i.values()))
			s = ss
		return s

	def select(self,u_from,u_item="*",u_where=None,u_limit=None,u_join=None,u_orderby=None,print_req=None):
		"""Selection dans la base de donnée"""
		if self.count(u_from,u_where,u_limit,u_join) > 0:
			s = self.build_order("SELECT",u_from,u_item,u_where,u_limit,u_join,u_orderby)
			if print_req is not None: print(s)
			self.cursor.execute(s)
			s = self.cursor.fetchall()
			if isinstance(s,list) and len(s) > 0:
				if isinstance(s[0],tuple):
					if len(s[0]) == 1:
						ss = list()
						for i in s:
							ss.append(i[0])
						s = ss
			return s
		else:
			return None

	def insert(self,tbl_name,fields,values):
		s = "INSERT INTO " + tbl_name + "(" + ",".join(fields) + ") VALUES (" + ",".join(values) + ")"
		print(s)
		self.cursor.execute(s)
		self.conn.commit()

	def create(self,tbname,items):
		s = "CREATE TABLE IF NOT EXISTS " + tbname + " (" + \
		"id INTEGER PRIMARY KEY AUTOINCREMENT," + \
		items + \
		")"
		self.cursor.execute(s)
		self.conn.commit()

	def last_insert_id(self):
		return self.request("SELECT last_insert_rowid() as id")["id"]

	def update(self,tbname,fields,values,u_where):
		d = dict(zip(fields,values))
		s0 = "UPDATE " + tbname + " SET " + ','.join([i + "=" +j for i,j in d.items()]) + ' WHERE ' + u_where
		self.cursor.execute(s0)#l,str(u_value))
		self.conn.commit()

	def table_exist(self,tbname):
		s = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='" + tbname + "'"

	def count(self,u_from,u_where=None,u_limit=None,u_join=None):
		"""Selection dans la base de donnée"""
		u_item = ""
		s = self.build_order("SELECT count(*) as ct",u_from,u_item,u_where,u_limit,u_join)
		self.cursor.execute(s)
		return self.cursor.fetchone()["ct"]
		
	def delete(self,u_from,u_where=None):
		if self.count(u_from,u_where=u_where) == 1:
			s = self.build_order("DELETE",u_from=u_from,u_where=u_where,u_item="")
			self.cursor.execute(s)
			self.conn.commit()

	def activate_for_keys(self):
		self.cursor.execute("PRAGMA foreign_keys = ON")

	def check_if_exist(self,tbname,u_id,u_value):
		if isinstance(u_id,list):
			s = ""
			for i in range(len(u_id)):
				s += u_id[i] + "='" + str(u_value[i]) + "' AND "
			s = s[:-4]
			N = self.count(tbname,u_where=s)
		else:
			N = self.count(tbname,u_where=u_id + "=" + str(u_value))
		return N

	def format_array(self,data):
		while (isinstance(data, list) or isinstance(data, tuple)) and len(data) == 1: data = data[0]
		return data
