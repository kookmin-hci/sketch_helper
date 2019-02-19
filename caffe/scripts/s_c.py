f = open("../75000_Train_stroke.txt", 'r')

for i in range(34500000):
    line = f.readline()
    print "line", line
    cls = line.split(' ')[len(line.split(' '))-1]
    stroke = line.split("_")[2].split(".")[0]
    
    print stroke+'_'+cls[:-1]+".txt"

    temp = open( stroke+'_'+cls[:-1]+".txt",'a')
    temp.write(line) 
    temp.close()
f.close
