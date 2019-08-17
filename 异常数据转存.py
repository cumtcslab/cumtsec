import os
filelist = []
data_label = []
m = 0
#存放数据
total_data = 'D:/total_data/total_bad.txt'
total = open(total_data,'w')
data = os.listdir('D:/train_set')

#将数据与区间分开放入两个列表
for file in data:
    if 'label' in file:
        data_label.append(file)

    else:
        filelist.append(file)


for datafile in filelist:
    single_data = open('D:/train_set/'+datafile,'r')
    dataline = single_data.read()
    train = dataline.split(',')
    #找到数据与区间文件共有的标签
    same = datafile[11:15]
    
    for labelfile in data_label:
        
        if labelfile[17:21] == same:
            single_label = open('D:/train_set/'+labelfile,'r')

            label = single_label.read()
            labellist = label.split(',')

            lens = len(labellist)
            #end记录文本最后一个
            end = len(train)
            begin = 0
            i = 0

            #循环各个区间
            while i<lens :
                count = 0
                #判断异常数据区间是否大于500
                if (begin ==0 and (int(labellist[i])-begin)>500) or (int(labellist[i])-int(labellist[begin]))>500:
                    #每组数据添加标签
                    total.write('0')
                    total.write(' ')
                
                    if begin == 0:
                        poor = int(labellist[i])-begin
                        x = begin
                        y = int(labellist[i])
                        while x<=y:
                            count += 1
                            total.write(str(train[x]))
                            if count%500 != 0:
                                total.write(' ')
                            if count%500 == 0:
                                if poor - count<500:
                                    total.write('\n')
                                    break
                                else:
                                    total.write('\n')
                                    total.write('0')
                                    total.write(' ')
                                    
                            x += 1

                    else:
                        poor = int(labellist[i])-int(labellist[begin])
                        x = int(labellist[begin])
                        y = int(labellist[i])
                        while x<=y:
                            count += 1
                            total.write(str(train[x]))
                            if count%500 != 0:
                                total.write(' ')
                            if count%500 == 0:
                                if poor - count<500:
                                    total.write('\n')
                                    break
                                else:
                                    total.write('\n')
                                    total.write('0')
                                    total.write(' ')
                                    
                            x += 1

                    
                        
                begin = i+1
                i += 2
            count = 0
            if end-int(labellist[lens-1])>500:
                   total.write('0')
                   total.write(' ')
                   poor = int(end)-int(labellist[lens-1])
                   x = int(labellist[lens-1])
                   y = int(end)
                   while x<y:
                       count += 1
                       total.write(str(train[x]))
                       if count%500 != 0:
                          total.write(' ')
                       if count%500 == 0:
                          if poor - count<500:
                              total.write('\n')
                              break
                          else:
                              total.write('\n')
                              total.write('0')
                              total.write(' ')
                                    
                       x += 1
                
            single_label.close()
    single_data.close()

total.close()
            
    
