'''
Description: 
2022-04-13 22:37:18
Created by Kai Zhou.
Email address is kz4yolo@gmail.com.
'''
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, roc_curve

def get_confusion_matrix(output,target):
    confusion_matrix = np.zeros((output[0].shape[0],output[0].shape[0]))
    for i in range(len(output)):
        true_idx = target[i]
        pred_idx = np.argmax(output[i])
        confusion_matrix[true_idx][pred_idx] += 1.0
    return confusion_matrix

def get_confusion_matrix_logits(output,target):
    confusion_matrix = np.zeros((2,2))
    for i in range(len(output)):
        true_idx = target[i]
        pred_idx = 1 if output[i]>0.5 else 0
        confusion_matrix[true_idx][pred_idx] += 1.0
    return confusion_matrix

def get_recall(confusion_matrix):
    recall = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        recall[i] = confusion_matrix[i][i]/confusion_matrix[i, :].sum()

    return recall

def get_precision(confusion_matrix):
    percision = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        percision[i] = confusion_matrix[i][i]/confusion_matrix[:, i].sum()

    return percision

def get_F1(confusion_matrix):
    pass 

def get_mul_cls_auc(output, target):
    one_hot_target = []
    
    for l in target:
        one_hot = np.zeros((output[0].shape[-1],))
        one_hot[int(l)] = 1
        one_hot_target.append(one_hot)
    target = np.asarray(one_hot_target)
    output = np.asarray(output)
    assert target.shape == output.shape, 'wrong shape! target shape:{}, ouput shape:{}'.format(target.shape, output.shape)
    
    return roc_auc_score(target, output, average='macro')
def get_binary_cls_auc(ouput, target):
    return roc_auc_score(target, output)

def get_each_auc(output, target):
    one_hot_target = []
    
    for l in target:
        one_hot = np.zeros((output[0].shape[-1],))
        one_hot[int(l)] = 1
        one_hot_target.append(one_hot)
    target = np.asarray(one_hot_target)
    auc_list = []
    output = np.asarray(output)
    assert target.shape == output.shape, 'wrong shape! target shape:{}, ouput shape:{}'.format(target.shape, output.shape)
    for i in range(output.shape[-1]):
        auc_list.append(roc_auc_score(target[:,i], output[:,i]))
        
    return np.asarray(auc_list)
        
