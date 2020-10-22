from sklearn.metrics import confusion_matrix
import numpy as np

def Frequency_Weighted_Intersection_over_Union(y_true, y_pred):
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
    freq = np.sum(cm,axis=1)/np.sum(cm)
    iu = np.diag(cm)/(np.sum(cm,axis=1)+np.sum(cm, axis=0)- np.diag(cm))
    FWIoU = (freq[freq>0]*iu[freq>0]).sum()
    return FWIoU