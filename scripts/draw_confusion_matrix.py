from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib


# example: baseline, labelled, task-aware
y_true = ['0']*979+['0']*1+['0']*15+['0']*5+['0']*0+['1']*3+['1']*993+['1']*2+['1']*0+['1']*2+['2']*8+['2']*0+['2']*960+['2']*18+['2']*14+['3']*20+['3']*2+['3']*16+['3']*952+['3']*10+['4']*3+['4']*0+['4']*6+['4']*19+['4']*972
y_pred = ['0']*979+['1']*1+['2']*15+['3']*5+['4']*0+['0']*3+['1']*993+['2']*2+['3']*0+['4']*2+['0']*8+['1']*0+['2']*960+['3']*18+['4']*14+['0']*20+['1']*2+['2']*16+['3']*952+['4']*10+['0']*3+['1']*0+['2']*6+['3']*19+['4']*972


norm = matplotlib.colors.Normalize(vmin=0, vmax=250)
C = confusion_matrix(y_true, y_pred, labels=['0', '1', '2', '3', '4'])
# C = confusion_matrix(y_true, y_pred, labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.matshow(C, cmap=plt.cm.GnBu, norm=norm)
cb = plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        if i == j:
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', color='white', family='Times New Roman', weight='bold', fontsize=13)
        else:
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', color='black', family='Times New Roman', weight='bold', fontsize=13)

plt.ylabel('Ground Truth', fontweight='bold', fontdict={'family': 'Times New Roman', 'size': 16})
plt.xlabel('Prediction', fontweight='bold', fontdict={'family': 'Times New Roman', 'size': 16})
cb.ax.tick_params(labelsize=13)
plt.tick_params(labelsize=16)
plt.xticks(range(0, 5), family='Times New Roman', weight='bold', labels=['0', '1', '2', '3', '4'])
plt.yticks(range(0, 5), family='Times New Roman', weight='bold', labels=['0', '1', '2', '3', '4'])
# plt.xticks(range(0, 10), family='Times New Roman', weight='bold', labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
# plt.yticks(range(0, 10), family='Times New Roman', weight='bold', labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# plt.show()
plt.savefig('./baseline_aware.pdf')
