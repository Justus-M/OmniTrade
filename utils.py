from sklearn.preprocessing import OneHotEncoder
import pickle
from pathlib import Path
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from seaborn import heatmap


def smote(x, y):
    oversample = SVMSMOTE()
    x, y = oversample.fit_resample(x, y)
    return x, y


def onehot(labels):
    encoder = OneHotEncoder()
    encoder.fit(labels.reshape(-1, 1))
    return encoder.transform(labels.reshape(-1, 1)).toarray()


def pickle_save(obj, fname):
    with open(f'{fname}.pickle', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(fpath):
    with open(fpath, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def roc(preds, true, legend='data',n = 0):
    if type(true[0]) in [np.ndarray, list]:
        true = np.argmax(true, axis=1)
    if type(preds[0])  in [np.ndarray, list]:
        preds = [p[1] for p in preds]
    fpr, tpr, thresh = metrics.roc_curve(true, preds)
    auc = round(metrics.roc_auc_score(true, preds), 3)
    plt.figure(n)
    plt.plot(fpr, tpr, label=f'{legend} - AUC: {auc}')
    plt.title('ROC Curve')
    plt.margins(0)
    plt.grid(which='major')
    plt.grid(b=True, which='minor', linestyle='--')
    plt.minorticks_on()
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.legend()

def heatmapper(preds, true, title='confusion matrix', labels = ['negative', 'positive'], n = 0):
    if len(np.squeeze(np.array(preds)).shape) > 1:
        preds = np.argmax(preds, axis=1)
    if len(np.squeeze(np.array(true)).shape) > 1:
        true = np.argmax(true, axis=1)

    true_labels, pred_labels = [], []
    for label in labels:
        pred_labels.append('predicted ' + label)
        true_labels.append('truth ' + label)
    plt.figure(n)
    cm = heatmap(confusion_matrix(true, preds), annot=True, cmap=plt.cm.Blues, fmt='g')
    cm.set_title(title)
    cm.set_xticklabels(pred_labels, rotation = 20)
    cm.set_yticklabels(true_labels, rotation = 90)

def filepaths(path, extension='*'):
    return [str(p.absolute()) for p in list(Path(path).rglob(f"*.{extension}"))]