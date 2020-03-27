# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import geopandas as gp
import numpy as np
import matplotlib
import csv

house_dic={}
sell_dic={}
area=[]
year_avg={}
vertex={}
city=["A","D","E","H","F"]

for year in range(104,109):
	year_avg[year]={}
	for season in range(1,5):
		if year==108 and season==4: break
		name="./data/"+str(year)+"S"+str(season)+".csv"
		print(name)
		with open(name, mode='r', encoding="UTF-8-sig") as file:
			rows = csv.reader(file)
			title=next(rows)
			next(rows)
			j=0
			for row in rows:
				temp={}
				if row[0] not in area:
					print(row[0])
					area.append(row[0])
				for i in range(len(title)-1):
					temp[title[i]]=row[i]
				if temp['交易標的'] == "土地" : continue
				try:
					if temp['單價元/平方公尺'] == "" : continue
				except:
					if temp['單價元平方公尺'] == "" : continue
				if temp['備註'] != "" : continue
				if '其他' in temp['都市土地使用分區']: continue
				if '農' in temp['非都市土地使用分區']: continue
				if '鄉' in temp['非都市土地使用分區']: continue
				if '山' in temp['非都市土地使用分區']: continue
				if temp['都市土地使用分區']=='住' : house_dic[row[-1]]=temp
				# house_dic[row[-1]]=temp
				if temp['都市土地使用分區']=='商' : sell_dic[row[-1]]=temp
				j+=1
				# if j == 10: break
		location_house={}
		area_avg={}
		for i in area:
			location_house[i]={}
			vertex[i]=[]
		for i in house_dic:
			location_house[house_dic[i]['鄉鎮市區']][i]=house_dic[i]
		for i in location_house:
			try:
				count=0
				summ=0
				for j in location_house[i]:
					count+=1
					try:
						summ+=int(location_house[i][j]['單價元/平方公尺'])
					except:
						summ+=int(location_house[i][j]['單價元平方公尺'])
				# print(i)
				# print(summ)
				# print(count)
				area_avg[i]=round(summ/count,2)
			except:
				continue
				# break
			# break
		year_avg[year][season]=area_avg
for i in year_avg:
	# sea=0
	for j in year_avg[i]:
		# if j==1: sea=0:
		# if j==2: sea=3:
		# if j==3: sea=6:
		# if j==4: sea=9:
		for k in year_avg[i][j]:
			vertex[k].append((i+0.1*3*(j-1),year_avg[i][j][k]))
print(vertex)
# target=['南區', '西區', '東區', '北屯區', '北區', '西屯區', '南屯區',
 # '豐原區', '后里區', '大甲區','沙鹿區','烏日區', '大里區', '霧峰區']
# target=[
# '北屯區','神岡區','外埔區','潭子區','太平區','新社區','大雅區']
target=[
'中區','大安區','龍井區','大肚區','梧棲區','石岡區',
'東勢區','清水區','和平區']
color=['#0DD145','#FFFF00','#FF00FF','#00FFFF','#0000FF','#00FF00','#FF0000','#000000',
'#F00B0D','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0','#7FFF00','#D2691E','#FF7F50',
'#6495ED','#FFF8DC','#DC143C','#00FFFF','#00008B','#008B8B','#B8860B','#BAA60B','#A8C6EB',
'#B8321B','#B1230B','#B08DFB','#B0987B','#B587FB','#6B64EB']


# co=0
plt.figure(figsize= (24.0, 12.0))
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
# for i in target:
# 	x = [p[0] for p in vertex[i]]
# 	y = [p[1] for p in vertex[i]]
# 	plt.plot(x, y, '-o',color=color[co],label=i)
# 	co+=1

# chinese =matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/kaiu.ttf')
# plt.legend(prop=chinese,loc='upper left')
# plt.savefig('line')
# plt.show()

villages_shp = gp.read_file("VILLAGE_MOI_121_1081121.shp", encoding="UTF-8-sig") #全台灣村里界圖
taichung_shp = villages_shp.query('COUNTYNAME=="臺中市"')
dummy_number=[]
for i in taichung_shp['TOWNNAME'].index.tolist():
	towname = taichung_shp.loc[i,"TOWNNAME"]
	try:
		print(towname,' ',(vertex[towname][-1][1])/vertex[towname][0][1])
		# dummy_number.append((vertex[towname][-1][1])/vertex[towname][0][1])
		dummy_number.append((vertex[towname][-1][1]))
	except:
		print(towname)
		dummy_number.append(0)
	# print(vertex[taichung_shp.loc[i,"TOWNNAME"]])
	# exit()))
taichung_shp['dummy number'] = np.array(dummy_number)
print(taichung_shp.head())

taichung_shp.plot(
    cmap=plt.cm.Reds, #指定顏色
    column='dummy number', #指定從自身的這個 column 讀取顏色深度
    legend=True
)
plt.savefig('temperature')
plt.show()