# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import matplotlib
import csv
year_avg={"A":[],"D":[],"E":[],"H":[],"F":[],"C":[]}
city=["A","D","E","H","F"]
# city=["A"]

for year in range(104,109):
	# year_avg[year]={}
	for season in range(1,5):
		if year==108 and season==4: break
		for ci in city:
			name="./5city/"+str(year)+"S"+str(season)+"/"+ci+"_lvr_land_A.csv"
			print(name)
			with open(name, mode='r', encoding="UTF-8-sig") as file:
				rows = csv.reader(file)
				title=next(rows)
				next(rows)
				t=0
				count=0
				summ=0
				for i in rows:
					try:
						summ+=int(i[22])
						count+=1
					except:
						continue
					t+=1
					if t ==1000:
						break
				year_avg[ci].append( (year+0.1*3*(season-1),round(summ/count,2)))
for year in range(104,109):
	year_avg[year]={}
	for season in range(1,5):
		if year==108 and season==4: break
		name="./data/"+str(year)+"s"+str(season)+".csv"
		print(name)
		with open(name, mode='r', encoding="UTF-8-sig") as file:
			rows = csv.reader(file)
			title=next(rows)
			next(rows)
			t=0
			count=0
			summ=0
			for i in rows:
				try:
					summ+=int(i[22])
					count+=1
				except:
					continue
				t+=1
				if t ==1000:
					break
			year_avg["C"].append( (year+0.1*3*(season-1),round(summ/count,2)))
print(year_avg)
plt.figure(figsize= (12.0, 6.0))
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
for i in year_avg:
	x = [p[0] for p in year_avg[i]]
	y = [p[1] for p in year_avg[i]]
	name=""
	if i =="A": name="台北市"
	if i =="D": name="台南市"
	if i =="E": name="高雄市"
	if i =="F": name="新北市"
	if i =="H": name="桃園市"
	if i =="C": name="台中市"
	plt.plot(x, y, '-o',label=name)
chinese =matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/kaiu.ttf')
plt.legend(prop=chinese,loc='best')
plt.savefig('ss')
plt.show()