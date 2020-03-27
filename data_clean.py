# -*- coding: utf-8 -*-
import csv
import tensorflow as tf
import numpy as np
import logging
import os
def getNumofBuildings_vector(string):
    index1=string.find("建物")
    index2=string.find("車位")
    return [int(string[2:index1]),int(string[index1+2:index2]),int(string[index2+2:]) ]
def loadData():
    with open('a_lvr_land_a.csv', 'r',encoding = 'UTF-8-sig') as csvfile:
        rows = csv.reader(csvfile)
        next(rows) #jump  2rows of head
        next(rows)
        trainData=[]
        x=0
        Township=["文山區","中正區","萬華區","大同區","內湖區","中山區","信義區","松山區","南港區","士林區","北投區","大安區"]
        use=["住","商","工","其他"]
        buildtype=['工廠','廠辦','店面(店鋪)','辦公商業大樓','透天厝','套房(1房1廳1衛)','公寓(5樓含以下無電梯)','華廈(10層含以下有電梯)','住宅大樓(11層含以上有電梯)','其他']
        street=[]
        for i in rows:
            # try:
                if i[1] == '土地':
                    continue
                ind=i[2].index('區')
                if i[2].find('街')!=-1:
                    t=i[2][ind+1:i[2].index('街')+1]
                elif i[2].find('路')!=-1:
                    t=i[2][ind+1:i[2].index('路')+1]
                if t not in street:
                    street.append(t)
                delete=[1,2,5,6,7,9,10,12,13,22,23,26,27]
                temp=[]
                Datatemp=[]
                for index in range(0,len(i)): #delete useless feature
                    if index not in delete:
                        temp.append(i[index])

                Township_vector=[0]*12              #鄉鎮市區
                Township_vector[Township.index(temp[0])]=1000000000
                Datatemp.append(np.array(Township_vector))

                vector=[0]*464
                vector[street.index(t)]=10000000000
                Datatemp.append(np.array(vector))
                # Use_vector=[0]*4                    #都市土地使用分區
                # Use_vector[use.index(temp[2])]=1    #這個會有沒分區的 直接過濾掉
                # Datatemp.append(np.array(Use_vector))

                NumofBuildings=temp[3]              #交易筆棟數
                NumofBuildings_vector=getNumofBuildings_vector(NumofBuildings)
                Datatemp.append(np.array(NumofBuildings_vector))

                Buildtype_vector=[0]*10             #建物型態
                Buildtype_vector[buildtype.index(temp[4])]=1000000000
                Datatemp.append(np.array(Buildtype_vector))

                feature_temp=[]
                feature_temp.append(float(temp[1])) #土地移轉總面積平方公尺
                # try:                                #建築完成年月
                    # feature_temp.append(float(temp[5]))
                # except:
                    # feature_temp.append(float(-1))
                if(float(temp[6])==0):
                    continue
                feature_temp.append(float(temp[6])) #建物移轉總面積平方公尺
                # feature_temp.append(float(temp[7])) #建物現況格局-房
                # feature_temp.append(float(temp[8])) #建物現況格局-廳
                # feature_temp.append(float(temp[9])) #建物現況格局-衛
                # if temp[10]=="有":                   #建物現況格局-隔間
                    # feature_temp.append(float(1))
                # else:
                    # feature_temp.append(float(-1))
                # if temp[11]=="有":                   #有無管理組織
                    # feature_temp.append(float(1))
                # else:
                    # feature_temp.append(float(-1))
                feature_temp.append(float(temp[13]))    #車位移轉總面積平方公尺
                Datatemp.append(np.array(feature_temp))
                Datatemp.append(np.array([np.array([float(temp[12])]),np.array([float(temp[14])])]))#房屋總價元,車位總價元
                trainData.append(Datatemp)

                # feature_temp.append(float(temp[12])-float(temp[14]))    #房屋總價元
                # feature_temp.append(float(temp[14]))    #車位總價元
            # except:
            #     continue
            # x+=1
            # if x==10:
            #     break
        print(np.array(trainData).shape)
        # print(len(street))
        return np.array(trainData)
        # for i in trainData:
        #   print(len(i))
        
class Batcher():
    def __init__(self, data, batchSize):
        self.__batchSize = batchSize
        suffled_data=self.shuffle(data)
        percentage=int(0.8*len(data))
        self.__train_data=suffled_data[:percentage]
        self.__validation_data=suffled_data[percentage:]
        self.__currentIdx = 0
        self.__epochNumber = 0
    def shuffle(self,data):
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        suffled_data = data[perm]
        return suffled_data
    def nextBatch(self):
        nextIdx = self.__currentIdx + self.__batchSize
        # print(nextIdx)
        # print(nextIdx)
        if nextIdx > len(self.__train_data):
            nextIdx = self.__batchSize
            self.__currentIdx = 0
            self.__train_data=self.shuffle(self.__train_data)
            self.__epochNumber += 1
        data = self.__train_data[self.__currentIdx:nextIdx]
        self.__currentIdx = nextIdx
        return self.reform(data)
    def reform(self,data):
        township = []
        use = []
        numofBuildings = []
        buildtype = []
        other_feature = []
        y_bulid = []
        y_car = []
        for i in data:
            township.append(i[0])
            use.append(i[1])
            numofBuildings.append(i[2])
            buildtype.append(i[3])
            other_feature.append(i[4])
            y_bulid.append(i[5][0])
            y_car.append(i[5][1])
        X=[np.array(township),np.array(use),np.array(numofBuildings),np.array(buildtype),np.array(other_feature)]
        Y=[np.array(y_bulid),np.array(y_car)]
        return X,Y
    def getValidationData(self):
        return self.reform(self.__validation_data)
    def getEpochNumber(self):
        return self.__epochNumber

class DNN(object):
    def __init__(self,sess):
        self.sess=sess
        y_bulid=tf.placeholder(tf.float32, [None, 1])
        y_car=tf.placeholder(tf.float32, [None, 1])
        keep_prob = tf.placeholder(tf.float32)

        Township=tf.placeholder(tf.float32, [None, 12])
        # township=tf.reshape(township,[-1,10*35*32])
        # t_dense=tf.layers.dense(inputs=Township, units=256, activation=tf.nn.relu)
        # t_dropout=tf.layers.dropout(inputs=t_dense, rate=keep_prob)
        # t_dense2=tf.layers.dense(inputs=t_dropout, units=256, activation=tf.nn.relu)
        # t_dropout2=tf.layers.dropout(inputs=t_dense2, rate=keep_prob)
        # t_out=tf.layers.dense(inputs=t_dropout2, units=1)

        Use=tf.placeholder(tf.float32, [None, 464])
        # use=tf.reshape(use,[-1,10*35*32])
        # u_dense=tf.layers.dense(inputs=Use, units=512, activation=tf.nn.relu)
        # u_dropout=tf.layers.dropout(inputs=u_dense, rate=keep_prob)
        # u_dense2=tf.layers.dense(inputs=u_dropout, units=512, activation=tf.nn.relu)
        # u_dropout2=tf.layers.dropout(inputs=u_dense2, rate=keep_prob)
        # u_out=tf.layers.dense(inputs=u_dropout2, units=1)

        NumofBuildings=tf.placeholder(tf.float32, [None, 3])
        # numofBuildings=tf.reshape(numofBuildings,[-1,10*35*32])
        # n_dense=tf.layers.dense(inputs=NumofBuildings, units=512, activation=tf.nn.relu)
        # n_dropout=tf.layers.dropout(inputs=n_dense, rate=keep_prob)
        # n_dense2=tf.layers.dense(inputs=n_dropout, units=512, activation=tf.nn.relu)
        # n_dropout2=tf.layers.dropout(inputs=n_dense2, rate=keep_prob)
        # n_out=tf.layers.dense(inputs=n_dropout2, units=1)

        Buildtype=tf.placeholder(tf.float32, [None, 10])
        # buildtype=tf.reshape(buildtype,[-1,10*35*32])
        # b_dense=tf.layers.dense(inputs=Buildtype, units=512, activation=tf.nn.relu)
        # b_dropout=tf.layers.dropout(inputs=b_dense, rate=keep_prob)
        # b_dense2=tf.layers.dense(inputs=b_dropout, units=512, activation=tf.nn.relu)
        # b_dropout2=tf.layers.dropout(inputs=b_dense2, rate=keep_prob)
        # b_out=tf.layers.dense(inputs=b_dropout2, units=1)

        Other_feature=tf.placeholder(tf.float32, [None, 3])
        # other_dense=tf.layers.dense(inputs=Other_feature, units=512, activation=tf.nn.relu)
        # other_dropout=tf.layers.dropout(inputs=other_dense, rate=keep_prob)
        # other_dense2=tf.layers.dense(inputs=other_dropout, units=512, activation=tf.nn.relu)
        # other_dropout2=tf.layers.dropout(inputs=other_dense2, rate=keep_prob)
        # other_out=tf.layers.dense(inputs=other_dropout2, units=1)

        # Concat=tf.concat([t_out, u_out, n_out, b_out, other_out], 1)
        Concat=tf.concat([Township,Use,Buildtype,Other_feature],1)
        build_dense1=tf.layers.dense(inputs=Concat, units=1024, activation=tf.nn.relu)
        build_dropout1=tf.layers.dropout(inputs=build_dense1, rate=keep_prob)
        build_dense2=tf.layers.dense(inputs=build_dropout1, units=1024, activation=tf.nn.relu)
        build_dropout2=tf.layers.dropout(inputs=build_dense2, rate=keep_prob)
        build_dense3=tf.layers.dense(inputs=build_dropout2, units=512, activation=tf.nn.relu)
        build_dropout3=tf.layers.dropout(inputs=build_dense3, rate=keep_prob)
        build_dense4=tf.layers.dense(inputs=build_dropout3, units=256, activation=tf.nn.relu)
        build_dropout4=tf.layers.dropout(inputs=build_dense4, rate=keep_prob)
        build_out=tf.layers.dense(inputs=build_dropout4, units=1)

        car_dense1=tf.layers.dense(inputs=Concat, units=512, activation=tf.nn.relu)
        car_dropout1=tf.layers.dropout(inputs=car_dense1, rate=keep_prob)
        car_dense2=tf.layers.dense(inputs=car_dropout1, units=256, activation=tf.nn.relu)
        car_dropout2=tf.layers.dropout(inputs=car_dense2, rate=keep_prob)
        car_out=tf.layers.dense(inputs=car_dropout2, units=1)

        # loss_build = tf.losses.absolute_difference(labels=y_bulid,predictions=build_out)
        loss_build = tf.log(tf.reduce_mean(tf.abs(tf.divide(tf.subtract(y_bulid,build_out),y_bulid))))
        # loss_build=tf.log(abs((y_bulid-build_out)/y_bulid))
        loss_car = tf.losses.absolute_difference(labels=y_car,predictions=car_out)
        optimizer=tf.train.AdamOptimizer(0.0000001).minimize(loss_build)

        self.Township = Township
        self.Use = Use
        self.NumofBuildings = NumofBuildings
        self.Buildtype = Buildtype
        self.Other_feature = Other_feature
        self.y_bulid = y_bulid
        self.y_car = y_car
        self.car_out = car_out
        self.loss_build = loss_build
        self.loss_car = loss_car
        self.optimizer = optimizer
        self.keep_prob = keep_prob

    def train(self, x, y, keepProb=1.0):
        self.sess.run([self.optimizer],feed_dict={
            self.Township : x[0],
            self.Use : x[1],
            self.NumofBuildings : x[2],
            self.Buildtype : x[3],
            self.Other_feature : x[4],
            self.y_bulid : y[0],
            self.y_car : y[1],
            self.keep_prob: keepProb,
        })
    def getAccuracy(self, x, y):
        return self.sess.run([self.loss_build, self.loss_car],feed_dict={
            self.Township : x[0],
            self.Use : x[1],
            self.NumofBuildings : x[2],
            self.Buildtype : x[3],
            self.Other_feature : x[4],
            self.y_bulid : y[0],
            self.y_car : y[1],
            self.keep_prob: 1.0,
        })
    def predict(self, x):
        return self.sess.run([self.car_out],feed_dict={
            self.Township : x[0],
            self.Use : x[1],
            self.NumofBuildings : x[2],
            self.Buildtype : x[3],
            self.Other_feature : x[4],
            self.keep_prob: 1.0,
        })
class Preditor(object):
    def __init__(self):
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess = tf.Session()
        self.__sess = sess
        self.__net = DNN(sess=sess)
    def train(self, numepoch=30000):
        init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.__sess.run(init_op)
        train_data = loadData()
        batcher = Batcher(data=train_data,batchSize=20)
        testX,testY=batcher.getValidationData()
        en=1
        numIterations=0
        flag=0
        while en < numepoch:
            numIterations+=1
            trainX,trainY=batcher.nextBatch()
            if en%1000==0 and flag==0:
                flag=1
                self.saveModel(str(en))
            if en%5!=0:
                flag=0
            if numIterations % 50 == 0:
                en = batcher.getEpochNumber()
                build_acc,car_acc = self.__net.getAccuracy(trainX, trainY)
                acc,dd = self.__net.getAccuracy(testX, testY)
                print('epoch %d, iteration %d, build_acc %f,vali %f ' % (en, numIterations, build_acc,acc))
                # print(acc)

                # predict=self.__net.predict(trainX)
                # print(trainX," ",trainY)
                # print(predict[0]," ",trainY[0][0])
            self.__net.train(trainX, trainY, keepProb=0.8)
    def saveModel(self, fileName):
        saver = tf.train.Saver()
        savePath = './model'+fileName+'/'
        mdlName = 'model'+fileName
        saver.save(self.__sess, os.path.join(savePath, mdlName))
        # print(X[0].shape)
        # print(X[1].shape)
        # print(X[2].shape)
        # print(X[3].shape)
        # print(X[4].shape)
preditor= Preditor()
preditor.train()
# a=loadData()