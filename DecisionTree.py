# coding=utf-8
import csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def getNumofBuildings_vector(string):
	index1=string.find("建物")
	index2=string.find("車位")
	return [string[2 : index1], string[index1 + 2 : index2], string[index2 + 2:]]

def getData(file):
	str_data = []
	scale_data = []
	a=0
	with open(file, newline = '', encoding = 'UTF-8-sig') as csvfile:
		rows = csv.DictReader(csvfile)
		for row in rows:
			new = []
			scale_new = []
			new.append(row['鄉鎮市區'])
			new.append(row['交易標的'])
			if row['都市土地使用分區'] == '':
				new.append('其他')
			else:
				new.append(row['都市土地使用分區'])
			trans_num = getNumofBuildings_vector(row['交易筆棟數'])
			scale_new.append(trans_num[0])##
			scale_new.append(trans_num[1])##
			scale_new.append(trans_num[2])##
			new.append(row['建物型態'])
			# if row['建築完成年月'] == '':
			# 	scale_new.append('0')
			# else:
			# 	build_year = row['建築完成年月'][:-4]
			# 	if len(build_year) >= 3 and build_year[0] == '0':
			# 		build_year = build_year[1:]
			# 	scale_new.append(build_year)
			scale_new.append(row['建物現況格局-房'])##
			scale_new.append(row['建物現況格局-廳'])##
			scale_new.append(row['建物現況格局-衛'])##
			new.append(row['建物現況格局-隔間'])
			new.append(row['有無管理組織'])
			scale_new.append(row['土地移轉總面積平方公尺'])##
			scale_new.append(row['建物移轉總面積平方公尺'])##
			new.append(row['總價元'])
			str_data.append(new)
			scale_data.append(scale_new)
			a+=1
			# if a==1:
				# break
	return str_data , scale_data
def onehot(feature):
	output=[]
	LABEL=['0_中區','0_北區','0_北屯區','0_南區','0_南屯區','0_后里區','0_和平區','0_外埔區',
	'0_大安區','0_大甲區','0_大肚區','0_大里區','0_大雅區','0_太平區','0_新社區','0_東勢區',
	'0_東區','0_梧棲區','0_沙鹿區','0_清水區','0_潭子區','0_烏日區','0_石岡區','0_神岡區',
	'0_西區','0_西屯區','0_豐原區','0_霧峰區','0_龍井區','1_土地','1_建物','1_房地(土地+建物)',
	'1_房地(土地+建物)+車位','1_車位','2_住','2_其他','2_商','2_工','2_農','3_住宅大樓(11層含以上有電梯)',
	'3_倉庫','3_公寓(5樓含以下無電梯)','3_其他','3_套房(1房1廳1衛)','3_工廠','3_店面(店鋪)',
	'3_廠辦','3_華廈(10層含以下有電梯)','3_辦公商業大樓','3_農舍','3_透天厝','4_有','4_無','5_有','5_無']

	for i in range(len(feature)):
		temp=[0]*len(LABEL)
		for j in range(len(feature[i])):
			temp[LABEL.index(str(j)+'_'+feature[i][j])]=1
		output.append(np.asarray(temp))
	print("fea",len(feature))
	print("out",len(output))
	return np.asarray(output)
if __name__ == '__main__':
	str_data,scale_data = getData('data.csv')
	str_data = np.asarray(str_data)
	scale_data = np.asarray(scale_data)
	scale_data = scale_data.astype(float)
	feature = np.asarray(str_data)[:,0:-1]
	label = np.asarray(str_data)[:,-1:]
	label = label.astype(float)
	# le = preprocessing.LabelEncoder()
	# le.fit(LABEL)
	# x= le.transform(str_data)
	# print(x)
	# print(label)
	# label = le.transform(label)
	# print(list(le.classes_))
	# print(label)
	df = pd.DataFrame.from_records(feature)
	# print(df)
	# print()
	# print(pd.get_dummies(df))
	input_data=pd.get_dummies(df)
	input_data=np.append(onehot(feature),scale_data,axis=1)
	print(len(input_data))
	# print(input_data)
	# print(df[0])
	# for i in x:
		# print(i)
	# for i in pd.get_dummies(df):
		# print(i)
	# print(input_data)
	regr = RandomForestRegressor(max_features=16,max_depth=15)
	# regr = DecisionTreeRegressor(max_depth=15)
	regr.fit(input_data,label)
	joblib.dump(regr, 'train.m')
	regr = joblib.load('train.m')

	vali_str,vali_scale = getData('./data/107s2.csv')
	vali_str = np.asarray(vali_str)
	vali_scale = np.asarray(vali_scale)
	# c=0
	# try:
	# 	for i in vali_scale:
	# 		c=i
	# 		i.astype(float)
	# except:
	# 	print("!!! ", c)
	vali_scale = vali_scale.astype(float)
	vali_feature = np.asarray(vali_str)[:,0:-1]
	vali_label = np.asarray(vali_str)[:,-1:]
	vali_label = vali_label.astype(float)
	vali_feature=onehot(vali_feature)
	vali_input=np.append(vali_feature,vali_scale,axis=1)
	print(len(vali_scale))
	print(len(vali_feature))
	print(len(vali_label))
	print(vali_input)
	print(regr.score(vali_input,vali_label))
	print(regr.score(input_data,label))

	# clf = tree.DecisionTreeClassifier(max_depth = 5, max_leaf_nodes=10000)
	# clf.fit(pd.get_dummies(df),label)
	# joblib.dump(clf, 'train.m')
	# clf.score(pd.get_dummies(df),label)

	# clf = joblib.load('train.m')
	# clf.predit(x)
	# print(feature[1])
	# print(label)
	# print(max(label))
	# print(len(label))

