import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
acc = np.load('SD/acc_0_5s.npy')

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(acc, density=False)
for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("acc", fontsize=16, labelpad = 15)
plt.ylabel("amount", fontsize=16, labelpad = 15)

plt.show()
data = np.load('SD/data.npy')
print('mean:', np.mean(data))
print('std:', np.std(data))
print('min:',  np.max(data))
print('max:',  np.min(data))

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(data, density=False)
#ax.bar(data, density=False)

for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Duration", fontsize=16, labelpad = 10)
plt.ylabel("Count", fontsize=16, labelpad = 10)
plt.show()


acc = np.load('SD/acc_1s.npy')
Kacc = np.load('SD/k_mean_acc_1s.npy')
fig, ax = plt.subplots()
df = pd.DataFrame(acc, columns=['Spectral'])
df2 = pd.DataFrame(Kacc, columns=['Kmeans'])

fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(df['Spectral'])
b_heights, b_bins = np.histogram(df2['Kmeans'], bins=a_bins)
width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='Spectral')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label='Kmeans')

for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Accuracy", fontsize=12, labelpad = 8)
plt.ylabel("Count", fontsize=12, labelpad = 8)
plt.legend(prop={'size': 10})
plt.show()

acc = np.load('SD/acc_0_5s.npy')
Kacc = np.load('SD/k_mean_acc_0_5s.npy')
fig, ax = plt.subplots()
df = pd.DataFrame(acc, columns=['Spectral'])
df2 = pd.DataFrame(Kacc, columns=['Kmeans'])

fig, ax = plt.subplots()
a_heights, a_bins = np.histogram(df['Spectral'])
b_heights, b_bins = np.histogram(df2['Kmeans'], bins=a_bins)
width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='Spectral')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label='Kmeans')

for rect in ax.patches:
    height = rect.get_height()
    ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Accuracy", fontsize=12, labelpad = 8)
plt.ylabel("Count", fontsize=12, labelpad = 8)
plt.legend(prop={'size': 10})
plt.show()

x = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0]
y = [0.61, 0.72, 0.82, 0.84, 0.89, 0.89, 0.88]
plt.figure(figsize=(15,10),dpi=100,linewidth = 2)
plt.plot(x, y,'s-',color = 'r', label="accuracy")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Segment duration", fontsize=12, labelpad = 8)
plt.ylabel("Accuracy", fontsize=12, labelpad = 8)

for idx in range(len(x)):
    show='('+str(x[idx])+', '+str(y[idx])+')'
    plt.annotate(show,xytext=(x[idx],y[idx]),xy=(x[idx],y[idx]))    

plt.legend(prop={'size': 10})
plt.show()



