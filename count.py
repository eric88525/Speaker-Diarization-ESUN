import numpy as np

fp = open('nohup.out', 'r')

lines = fp.readlines()
ele = []
for line in lines:
    if 'acc' in line:
        e = float(line.split()[-1])
        ele.append(e)

print(np.mean(ele), len(ele))

