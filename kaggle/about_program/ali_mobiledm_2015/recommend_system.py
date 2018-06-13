# This file realize the recommendation system
import pymysql
import pandas as pd

db = pymysql.connect("localhost", "root", "root", "j18_panjin_yonghuhuaxiang")
sql = "select * from ans_sample"


data = pd.read_sql(sql, db)