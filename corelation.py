import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
data = pd.read_csv('./data/107s2.csv', encoding='utf-8')
data.drop([ '交易標的', '土地區段位置/建物區段門牌', \
	 '非都市土地使用分區', '非都市土地使用編定',\
	 '交易筆棟數', '移轉層次', '總樓層數', '建物型態', '主要用途', '主要建材', \
	  '建物現況格局-隔間', '有無管理組織','總價元', \
	 '車位類別', '車位移轉總面積平方公尺', '車位總價元', '備註', '編號'], axis=1, inplace=True)
print(data.columns.tolist())
print(data)
for i in range(len(data)):
	if data.loc[i,'都市土地使用分區'] == '住':
		data.loc[i,'都市土地使用分區'] = 0
	elif data.loc[i,'都市土地使用分區'] == '商':
		data.loc[i,'都市土地使用分區'] = 1
	elif data.loc[i,'都市土地使用分區'] == '工':
		data.loc[i,'都市土地使用分區'] = 2
	else:
		data.loc[i,'都市土地使用分區'] = 3
location = []
for i in range(len(data)):
	if data.loc[i,'鄉鎮市區'] not in location:
		location.append(data.loc[i,'鄉鎮市區'])
for i in range(len(data)):
	data.loc[i,'鄉鎮市區'] = location.index(data.loc[i,'鄉鎮市區'])
	# exit()
print(data)
corrMatrix = data.corr()
# print(corrMatrix)
chinese =matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/kaiu.ttf')
plt.figure(figsize= (18.0, 18.0))
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
sn.set(font=chinese.get_name())
sn.heatmap(corrMatrix, annot=True)
plt.savefig('tt')
plt.show()