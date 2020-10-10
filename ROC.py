import glob
from PIL import Image
from utils import image_analysis

path = "./ROC"
species = ["Staphylococcus aureus", "Coagulase-negative Staphylococcus"]
#species = ["Coagulase-negative Staphylococcus"]

probs = dict.fromkeys(species)
for key in probs.keys():
    probs[key] = []

# Evaluate overall probability of each snapshot
for sp in species:
    files = glob.glob(path + "/" + sp + "/*.JPG")
    for filename in files:
        img = Image.open(filename)
        # Image classification
        crop_width, crop_height = 512, 512
        class_indices, result = image_analysis.image_classifier(img, crop_width, crop_height, verbose=False)

        # Calculate overall probability
#        overall_prob = image_analysis.calc_overall_probability(class_indices, result, max_method=True, max_key="Staphylococcus aureus")
        overall_prob = image_analysis.calc_overall_probability(class_indices, result)
        probs[sp].append(overall_prob)
        print("Evaluated {}".format(filename))

# ROC analysis
from sklearn import metrics
import matplotlib.pyplot as plt
target = "Staphylococcus aureus"
y_true = []
y_pred = []
for sp in probs.keys():
    for prob in probs[sp]:
        y_true.append(1 if sp == target else 0)
        y_pred.append(prob[target])

print(y_true)
print(y_pred)

# Plot distribution
import pandas as pd
df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})
x0 = df[df['y_true']==0]['y_pred']
x1 = df[df['y_true']==1]['y_pred']

fig = plt.figure(figsize=(6,5)) #
ax = fig.add_subplot(1, 1, 1)
ax.hist([x0, x1], bins=10, stacked=True)

plt.xticks(np.arange(0, 1.1, 0.1), fontsize = 13) #arangeを使うと軸ラベルは書きやすい
plt.yticks(np.arange(0, 6, 1), fontsize = 13)

plt.ylim(0, 4)
plt.show()

# ROC curve
fpr, tpr, thres = metrics.roc_curve(y_true, y_pred)
auc = metrics.auc(fpr, tpr)
print('auc:', auc)

plt.figure(figsize = (5, 5)) #単一グラフの場合のサイズ比の与え方
plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False Positive Rete', fontsize = 13)
plt.ylabel('TPR: True Positive Rete', fontsize = 13)
plt.grid()
plt.show()