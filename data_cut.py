
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
data_cut_name=r"D:\zhang yin\4.30\哈尔乌素煤矿_1025-pd_102_17_20190430165221.log"
data_name="D:/zhang yin/train_set/data_line9_17_6_20190430165221.txt"
label_name="D:/zhang yin/train_set/data_line9_label_17_6_20190430165221.txt"
inFile = open(data_cut_name,'r')
y = []
j=1
for line in inFile:                    #循环遍历每行文本数据
    j=j+1
    if j%10==0:                         #隔行读取
        train = line.split(',')        #将用逗号隔开的数据存放到列表中
        y.append(float(train[8]))      #将数据第九列加入空列表中
lens = len(y)     
data=open(data_name,"w")
#存数据
for i in range(lens):
    txt=str(y[i])
    data.write(txt)
    if i==lens-1:
        break
    data.write(",")
data.close()
inFile=open(data_name)
#读数据
data=inFile.read()
data=data.split(',')
lens=len(data)
print(lens)
print(data)
lens=len(data)
y=[]
rise=[]
down=[]
for i in range(lens):
    y.append(float(data[i]))
leny=len(y)
for i in range(leny-1):
    value=y[i+1]-y[i]
    #寻找跳变位置
    if value>=30000:
        if len(rise)==0:
            rise.append(i)
        #判断跳点的位置间距是否太小
        elif i-rise[len(rise)-1]>=4000:
            rise.append(i)
        
        else:
            pass
    if value<=-35000:
        if len(down)==0:
            down.append(i)
        #判断跳点的位置间距是否太小
        elif i-down[len(down)-1]>=4000:
            down.append(i)
        #间距太小否定上一个位置
        else:
            down.pop()
print(rise)
print(down)
a=[[0 for i in range(2)] for i in range(len(down))]
labelFile=open(label_name,"a")
for i in range(len(down)):
    
    for j in range(len(rise)):
        if rise[j]>down[i]:
            a[i][0]=down[i]
            a[i][1]=rise[j]
            print(down[i],rise[j],"is good",sep=",")
            if i==0:
                labelFile.write(str(down[i]))
                labelFile.write(",")
                labelFile.write(str(rise[j]))
            else:
                labelFile.write(",")
                labelFile.write(str(down[i]))
                labelFile.write(",")
                labelFile.write(str(rise[j]))
            break
labelFile.close()
print(a)


# In[ ]:




