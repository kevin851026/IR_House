# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
import geopandas as gp
import pandas as pd
import numpy as np
import matplotlib
import csv
from sklearn import linear_model
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

build_zoning_data = []
build_state_data = []
live_data = []
industury_data = []
sell_data = []
type_A_data = []
type_B_data = []
type_C_data = []
type_D_data = []
type_E_data = []
type_F_data = []
type_G_data = []
for year in range(104,109):
    for season in range(1,5):
        if year==108 and season==4: break
        name="./data/"+str(year)+"S"+str(season)+".csv"
        print(name)
        with open(name, mode='r', encoding="UTF-8-sig") as file:
            rows = csv.reader(file)
            title=next(rows)
            next(rows)
            live = 0
            live_price = 0
            industury = 0
            industury_price = 0
            sell = 0
            sell_price = 0
            type_A=0
            type_A_price=0
            type_B=0
            type_B_price=0
            type_C=0
            type_C_price=0
            type_D=0
            type_D_price=0
            type_E=0
            type_E_price=0
            type_F=0
            type_F_price=0
            type_G=0
            type_G_price=0
            for row in rows:
                try:
                    price =int(row[22])
                    if row[4] =='住':
                        live += 1
                        live_price += price
                        live_data.append(round(price/1000))
                    if row[4] =='工':
                        industury += 1
                        industury_price += price
                        industury_data.append(round(price/1000))
                    if row[4] =='商':
                        sell += 1
                        sell_price += price
                        sell_data.append(round(price/1000))
                except:
                    continue
                try:
                    price =int(row[22])
                    if row[11]=='公寓(5樓含以下無電梯)':
                        type_A += 1
                        type_A_price += int(price)
                        type_A_data.append(round(price/1000))
                    if row[11]=='華廈(10層含以下有電梯)':
                        type_B += 1
                        type_B_price += int(price)
                        type_B_data.append(round(price/1000))
                    if row[11]=='住宅大樓(11層含以上有電梯)':
                        type_C += 1
                        type_C_price += int(price)
                        type_C_data.append(round(price/1000))
                    if row[11]=='透天厝':
                        type_D += 1
                        type_D_price += int(price)
                        type_D_data.append(round(price/1000))
                    if row[11]=='店面(店鋪)':
                        type_E += 1
                        type_E_price += int(price)
                        type_E_data.append(round(price/1000))
                    if row[11]=='套房(1房1廳1衛)':
                        type_F += 1
                        type_F_price += int(price)
                        type_F_data.append(round(price/1000))
                    if row[11]=='辦公商業大樓':
                        type_G += 1
                        type_G_price += int(price)
                        type_G_data.append(round(price/1000))
                except:
                    continue
            build_zoning_data.append([year,season,live,industury,sell,live_price,industury_price,sell_price])
            build_state_data.append([year,season,type_A,type_B,type_C,type_D,type_E,type_F,type_G\
                                    ,type_A_price,type_B_price,type_C_price,type_D_price,type_E_price,type_F_price,type_G_price])
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
# plt.xticks(np.linspace(107,109,3))
# print(build_zoning_data)
chinese =matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/kaiu.ttf')
#--都市土地使用分區 交易數量---
# for build_zoning in range(2,5):
#   x_data = []
#   y_data = []
#   for i in build_zoning_data:
#       print(i)
#       x_data.append(i[0]+0.25*(i[1]-1)-0.05+0.05*(build_zoning-2))
#       y_data.append(i[build_zoning])
#   plt.bar(x_data,y_data,width = 0.05, label = ['住','工','商'][build_zoning-2])
# plt.legend(prop=chinese,loc= 'best')
# plt.show()

#--都市土地使用分區 價格趨勢---
# for build_zoning in range(2,5):
#   x_data = []
#   y_data = []
#   for i in build_zoning_data:
#       x_data.append(i[0]+0.25*(i[1]-1))
#       y_data.append(i[build_zoning+3]/i[build_zoning])
#   print(x_data)
#   print(y_data)
#   plt.plot(x_data,y_data,label = ['住','工','商'][build_zoning-2])
# plt.legend(prop=chinese,loc= 'best')
# plt.show()
#-----都市土地使用分區 平均價格分布-------
# live_data = [round(i/5)*5 for i in live_data if i <250]
# industury_data = [round(i/5)*5 for i in industury_data if i <250]
# sell_data = [round(i/5)*5 for i in sell_data if i <250]
# plt.hist(live_data, bins=50,histtype="stepfilled",alpha=.6,label='住')
# plt.hist(sell_data, bins=50,histtype="stepfilled", alpha=.6,label='商')
# plt.hist(industury_data, bins=50,histtype="stepfilled", alpha=1,label='工')
# # plt.yticks(np.linspace(0,250,26))
# plt.legend(prop=chinese,loc= 'best')
# plt.show()
#----建築房屋型態 交易數量------
# state=['公寓(5樓含以下無電梯)','華廈(10層含以下有電梯)','住宅大樓(11層含以上有電梯)',\
#         '透天厝','店面(店鋪)','套房(1房1廳1衛)','辦公商業大樓']
# for build_state in range(2,9):
#     x_data = []
#     y_data = []
#     for i in build_state_data:
#         print(i)
#         x_data.append(i[0]+0.25*(i[1]-1)-0.08+0.03*(build_state-2))
#         y_data.append(i[build_state])
#     plt.bar(x_data,y_data,width = 0.03, label = state[build_state-2])
# plt.legend(prop=chinese,loc= 'best')
# plt.show()

#--建築房屋型態 價格趨勢-------
# state=['公寓(5樓含以下無電梯)','華廈(10層含以下有電梯)','住宅大樓(11層含以上有電梯)',\
#       '透天厝','店面(店鋪)','套房(1房1廳1衛)','辦公商業大樓']
# for build_state in range(2,9):
#   x_data = []
#   y_data = []
#   for i in build_state_data:
#       x_data.append(i[0]+0.25*(i[1]-1))
#       y_data.append(i[build_state+7]/i[build_state])
#   x_data = np.array(x_data)
#   y_data = np.array(y_data)
#   b, m = polyfit(x_data, y_data, 1)
#   # plt.scatter(x_data, y_data,  marker='.') '點'
#   plt.plot(x_data, b + m*x_data, linestyle='-',label = state[build_state-2]) #'回歸線'
#   # plt.plot(x_data,y_data,label = state[build_state-2]) #原曲線
# plt.legend(prop=chinese,loc= 'best')
# plt.show()
#-----建築房屋型態 平均價格分布-------
type_A_data = [round(i/5)*5 for i in type_A_data if i <200]
type_B_data = [round(i/5)*5 for i in type_B_data if i <200]
type_C_data = [round(i/5)*5 for i in type_C_data if i <200]
type_D_data = [round(i/5)*5 for i in type_D_data if i <200]
type_E_data = [round(i/5)*5 for i in type_E_data if i <200]
type_F_data = [round(i/5)*5 for i in type_F_data if i <200]
type_G_data = [round(i/5)*5 for i in type_G_data if i <200]

plt.hist(type_C_data, bins=40,histtype="stepfilled",alpha=.6,label='住宅大樓(11層含以上有電梯)')
plt.hist(type_D_data, bins=40,histtype="stepfilled",alpha=.6,label='透天厝')
plt.hist(type_B_data, bins=40,histtype="stepfilled",alpha=1,label='華廈(10層含以下有電梯)')
plt.hist(type_F_data, bins=40,histtype="stepfilled",alpha=1,label='套房(1房1廳1衛)')
plt.hist(type_A_data, bins=40,histtype="stepfilled",alpha=1,label='公寓(5樓含以下無電梯)')
plt.hist(type_G_data, bins=40,histtype="stepfilled",alpha=1,label='辦公商業大樓')
plt.hist(type_E_data, bins=40,histtype="stepfilled",alpha=1,label='店面(店鋪)')

# plt.yticks(np.linspace(0,250,26))
plt.legend(prop=chinese,loc= 'best')
plt.show()