import os
import glob
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
#%matplotlib inline
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split  # 对数据集切分
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
csv_list = glob.glob('E:/课程作业/机器学习导引/sh/20201223/*.csv')
if os.path.exists('./result.csv'):  # 如果文件存在
    # 删除文件
    os.remove('./result.csv')  
print(u'共发现%s个CSV文件'% len(csv_list))
print(u'正在处理............')
for i in csv_list: #循环读取同文件夹下的csv文件
    fr = open(i,'rb').read()
    with open('result.csv','ab') as f: #将结果保存为result.csv
        f.write(fr)
print(u'合并完毕！')
#以gbk形式读取文件，同时为文件加上列名
houseprice = pd.read_csv('result.csv',encoding='gbk',header=None,names = ['时间','区','镇','基本情况','房价','房子构造','网页链接'])
print(houseprice.shape[0])
#查看缺失值总数
print((houseprice.isnull()).sum())
#删除至少包含4个非NaN值的行
houseprice.dropna(axis=0,thresh = 4, inplace = True)
#这种由于删除之后会造成索引值的空缺,所以重新设置索引值
houseprice = houseprice.reset_index(drop=True)
#print(len(houseprice['房子构造'][i]))计算正常分为的文本长度，大约48左右
#删掉房子构造不正确的数据
#经过观察，发现并不是所有的房子构造数据都可以变成七列，由于不能变成七列的只是少部分，删掉这些,
print((houseprice['房子构造'].str.split('|').map(len) == 7).sum())
for i in range(houseprice.shape[0]):
    if len(houseprice['房子构造'][i].split('|')) != 7:
        houseprice = houseprice.drop(i) 
#删除不符合条件的行数之后会再次出现缺失值，所以再次重新设置
houseprice = houseprice.reset_index(drop=True)
houseprice['房间数']= houseprice['房子构造'].map(lambda x:x.split('|')[0])
houseprice['面积']= houseprice['房子构造'].map(lambda x:x.split('|')[1])
houseprice['朝向']= houseprice['房子构造'].map(lambda x:x.split('|')[2])
houseprice['装修']= houseprice['房子构造'].map(lambda x:x.split('|')[3])
houseprice['楼层数']= houseprice['房子构造'].map(lambda x:x.split('|')[4])
houseprice['建造年份']= houseprice['房子构造'].map(lambda x:x.split('|')[5])
houseprice['楼房类型']= houseprice['房子构造'].map(lambda x:x.split('|')[6])
#观察发现有些数据没有建造年份这一项，删除没有这项数据的行，分别为暂无数据，板房,板塔结合
#print(houseprice['建造年份'].unique())#查看年份分类
for i in range(houseprice.shape[0]):
    if houseprice['建造年份'][i] == ' 板楼 ' or houseprice['建造年份'][i] == ' 板塔结合 ' or houseprice['建造年份'][i] ==' 暂无数据 ' :
        houseprice = houseprice.drop(i) 
houseprice = houseprice.reset_index(drop=True)
print(houseprice.shape[0])
#由于线性回归模型只能处理数据，所以我们要把数据转化为纯数字
houseprice['面积'] = houseprice['面积'].map(lambda x:x.split('平')[0])
houseprice['房价'] = houseprice['房价'].map(lambda x:x.split('万')[0])
#一室一厅按照两个房间计算
houseprice['房间数1'] = houseprice['房间数'].map(lambda x:x.split('室')[0])
houseprice['房间数2'] = houseprice['房间数'].map(lambda x:x.split('室')[1])
houseprice['房间数2'] = houseprice['房间数2'].map(lambda x:x.split('厅')[0])
#对表格数据类型进行强制转换
houseprice[['房间数1','房间数2']] = houseprice[['房间数1','房间数2']].astype('float')
#print(houseprice['房间数2'].dtype)
houseprice['房间数'] = houseprice['房间数1']+houseprice['房间数2']
pd.set_option('mode.chained_assignment', None)
#将装修类型转换为数字
houseprice['装修'].unique()#查看装修分类
for i in range(houseprice.shape[0]):
    if houseprice['装修'][i] == ' 精装 ':
        houseprice['装修'][i] = 1
    elif houseprice['装修'][i] == ' 简装 ':
        houseprice['装修'][i] = 2
    elif houseprice['装修'][i] == ' 毛坯 ':
        houseprice['装修'][i] = 3
    elif houseprice['装修'][i] == ' 其他 ':
        houseprice['装修'][i] = 4
#将建造年份变成单纯的数字
houseprice['建造年份'] = houseprice['建造年份'].map(lambda x:x.split('年')[0])
#将楼房类型转换为数字
houseprice['楼房类型'].unique()
for i in range(houseprice.shape[0]):
    if houseprice['楼房类型'][i] == ' 板楼':
        houseprice['楼房类型'][i] = 1
    elif houseprice['楼房类型'][i] == ' 塔楼':
        houseprice['楼房类型'][i] = 2
    elif houseprice['楼房类型'][i] == ' 板塔结合':
        houseprice['楼房类型'][i] = 3
    elif houseprice['楼房类型'][i] == ' 平房':
        houseprice['楼房类型'][i] = 4
    else:#暂无数据
        houseprice['楼房类型'][i] = 0
#将楼房类型转换为数字
houseprice['区'].unique()
for i in range(houseprice.shape[0]):
    if houseprice['区'][i] == '宝山':
        houseprice['区'][i] = 1
    elif houseprice['区'][i] == '长宁':
        houseprice['区'][i] = 2
    elif houseprice['区'][i] == '奉贤':
        houseprice['区'][i] = 3
    elif houseprice['区'][i] == '虹口':
        houseprice['区'][i] = 4
    elif houseprice['区'][i] == '黄浦':
        houseprice['区'][i] = 5
    elif houseprice['区'][i] == '嘉定':
        houseprice['区'][i] = 6
    elif houseprice['区'][i] == '静安':
        houseprice['区'][i] = 7
    elif houseprice['区'][i] == '金山':
        houseprice['区'][i] = 8
    elif houseprice['区'][i] == '闵行':
        houseprice['区'][i] = 9
    elif houseprice['区'][i] == '浦东':
        houseprice['区'][i] = 10
    elif houseprice['区'][i] == '普陀':
        houseprice['区'][i] = 11
    elif houseprice['区'][i] == '青浦':
        houseprice['区'][i] = 12
    elif houseprice['区'][i] == '松江':
        houseprice['区'][i] = 13
    elif houseprice['区'][i] == '徐汇':
        houseprice['区'][i] = 14
    elif houseprice['区'][i] == '杨浦':
        houseprice['区'][i] = 15
#对于朝向进行处理，朝南最好，朝北最差，其余均设为一个值
#首先对相关空格进行处理,发现只有一个字长度为3
for i in range(houseprice.shape[0]):
    if len(houseprice['朝向'][i].split(' ')) != 3:
        houseprice['朝向'][i] = houseprice['朝向'][i].split(' ')[0]
#进行数字化处理
houseprice['朝向'].unique()
for i in range(houseprice.shape[0]):
    if houseprice['朝向'][i] == ' 南 ':
        houseprice['朝向'][i] = 1
    elif houseprice['朝向'][i] == ' 北 ':
        houseprice['朝向'][i] = 2
    elif houseprice['朝向'][i] == ' 东 ' or houseprice['朝向'][i] == ' 西 ':
        houseprice['朝向'][i] = 3
    else:
        houseprice['朝向'][i] = 4
#对数据类型进行强行转换，方便接下来训练，不转换也可，最后只在y_true转换也可
houseprice[['房价','区','面积','朝向','装修','建造年份','楼房类型']] = houseprice[['房价','区','面积','朝向','装修','建造年份','楼房类型']].astype('float')
#print(houseprice['区'].dtype)
houseprice = houseprice[0:100]
#由于本数据是同一时刻爬取的数据，所以时间不是影响因素，‘基本情况’，‘网页链接’影响情况有限，而楼层数和小区数太过多样化，所以不考虑
train = houseprice.loc[:,['区','房间数','面积','朝向','装修','建造年份','楼房类型']]  # 样本
price = houseprice.房价  
X_train, x_test, y_train, y_true = train_test_split(train, price, test_size=0.2)
linear = LinearRegression()
linear.fit(X_train, y_train)
y_pre_linear = linear.predict(x_test)
linear_score = r2_score(y_true, y_pre_linear)
print(linear_score)
print(linear.intercept_)#截距
print(linear.coef_)#斜率
y_true = y_true.reset_index(drop=True)#对真实值重新排序
#y_true = y_true.astype('float')#对真实值进行强制类型转换，不然没办法变成图
plt.plot(y_true, label='true')
plt.plot(y_pre_linear, label='linear')
plt.legend()
if os.path.exists('./hp_predict.jpg'):  # 如果文件存在
    # 删除文件
    os.remove('./hp_predict.jpg') 
plt.savefig("hp_predict.jpg")
#plt.savefig("hp_predictall.jpg")
plt.show()




    
