import matplotlib.pyplot as plt

fp = open('vox1_SM_scores.txt')
VEER = []
TEER = []

line = fp.readline()
while line:
    if 'VEER' in line:
        e = float(line.split()[-1])
        VEER.append(e)
    elif 'TEER' in line:
        line = line.split(',')[1]
        e = float(line.split()[-1])
        TEER.append(100 - e)
    line = fp.readline()
    
print(VEER)
print(TEER)

fig, ax1 = plt.subplots()
plt.title('Error Rate: vox1, SM')
plt.xlabel('Epoches')
ax2 = ax1.twinx()
ax1.set_ylabel('Train Error Rate', color='tab:blue')
ax1.plot(range(len(TEER)), TEER, color='tab:blue', alpha=0.75, label="Train ERR")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2.set_ylabel('Validation Error Rate', color='black')
ax2.plot(range(len(VEER)), VEER, color='black', alpha=1, label="Val ERR")
ax2.tick_params(axis='y', labelcolor='black')

fig.tight_layout()
fig.legend(loc = 'best')

plt.show()
