f = open("shadow_robot_dataset.csv",'r')
out = open("splited_dataset.csv",'w')
for i in range(10000):
    line = f.readline()
    out.write(line)

f.close()
out.close()