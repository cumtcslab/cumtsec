import os
filelist = []
data_label = []
m = 0
total_data = 'D:/total_data/total[副本].txt' #
total = open(total_data,'w')
data = os.listdir('D:/train_set')
for file in data:
    if 'label' in file:
        data_label.append(file)

    else:
        filelist.append(file)


for datafile in filelist:
    single_data = open('D:/train_set/'+datafile,'r')
    dataline = single_data.read()
    train = dataline.split(',')    
    same = datafile[11:15]
    
    for labelfile in data_label:
        
        if labelfile[17:21] == same:
            single_label = open('D:/train_set/'+labelfile,'r')

            label = single_label.read()
            labellist = label.split(',')

            lens = len(labellist)
            
            i = 0
            num = 0
            j = 0
            while i<lens :
                num = 0
                poor = int(labellist[i+1])-int(labellist[i])
                y = int(labellist[i+1])
                x = int(labellist[i])
                total.write('1')
                total.write(' ')
                while x<= y:
                    num += 1
                    total.write(train[x])
                    if num%500 != 0:
                        total.write(' ')
                    if num%500 == 0 and poor-num>=500:
                        total.write('\n')
                        total.write('1')
                        total.write(' ')
                    elif num%500 ==0 and poor-num<500:
                        break

                    x += 1

                total.write('\n')
                i += 2

total.close()
single_data.close()
            
    
