import sqlite3 as sql
import re
import numpy
class Usql:
	"""
	Classe permettant de réaliser des opérations sur une base SQLite3
	"""
	cursor = None
	def __init__(self,curs,conn):
		"""Constructeur"""
		self.cursor = curs
		self.conn = conn
		self.cursor.execute("""SELECT name  FROM sqlite_master WHERE type = 'table'""")
		self.tbl_name = numpy.ravel(self.cursor.fetchall())
	
	def col_name(self,table_name):
		"""Récupérer la liste des noms de colonne d'une table"""
		# self.cursor.execute("SELECT sql  FROM sqlite_master WHERE type = 'table' AND name = '" + table_name +"'")
		c = self.cursor.execute("SELECT *  FROM " + table_name)
		tables = list(map(lambda x: x[0], c.description))
		return tables

	def build_order(self,u_order,u_from,u_item="*",u_where=None,u_limit=None,u_join=None,u_orderby=None):
		s =u_order + " " + u_item + " FROM " + u_from
		if u_join is not None:
			s += " " + u_join 
		if u_where is not None:
			s += " WHERE " + u_where
		if u_limit is not None:
			s += " LIMIT " + u_limit
		if u_orderby is not None:
			s += " ORDER BY " + u_orderby
		return s

	def select(self,u_from,u_item="*",u_where=None,u_limit=None,u_join=None,u_orderby=None):
		"""Selection dans la base de donnée"""
		if self.count(u_from,u_where,u_limit,u_join) > 0:
			s = self.build_order("SELECT",u_from,u_item,u_where,u_limit,u_join,u_orderby)
			# print(s)
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
		s0 = '''INSERT INTO ''' + tbl_name + '''('''
		s00 = ""
		for i in fields:
			s0 += i + ''','''
			s00 += "?," 
		s0 = s0[:-1]
		s0 += ") "
		s00 = s00[:-1]
		s00 = "VALUES(" + s00 + ")"
		s0 += s00 
		s1 = tuple(values)
		self.cursor.execute(s0,s1)
		self.conn.commit()

	def create(self,tbname,items):
		s = "CREATE TABLE IF NOT EXISTS " + tbname + " (" + \
		"id INTEGER PRIMARY KEY AUTOINCREMENT," + \
		items + \
		")"
		self.cursor.execute(s)
		self.conn.commit()


	def update(self,tbname,u_field,u_id,u_value):
		while isinstance(u_id,int) is not True and isinstance(u_id,str) is not True:
			u_id = u_id[0]
		s0 = 'UPDATE ' + tbname + ' SET ' + u_field + ' = ' + str(u_value) + ' WHERE id = ' + str(u_id)
		self.cursor.execute(s0)#l,str(u_value))
		self.conn.commit()

	def table_exist(self,tbname):
		s = "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='" + tbname + "'"

	def count(self,u_from,u_where=None,u_limit=None,u_join=None):
		"""Selection dans la base de donnée"""
		u_item = ""
		s = self.build_order("SELECT count(*)",u_from,u_item,u_where,u_limit,u_join)
		# print(s)
		self.cursor.execute(s)
		return self.cursor.fetchone()[0]
		
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