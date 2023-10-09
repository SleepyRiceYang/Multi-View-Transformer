import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.stats import norm
import pandas as pd

def cal_auc_ci_i(predictions_prob, true_labels, ci=0.95):
    n = len(predictions_prob)
    auc = roc_auc_score(true_labels, predictions_prob)
    auc_var = (2*auc -auc**2) / (n-1)
    auc_std = np.sqrt(auc_var)
    z_score = norm.ppf(1-(1-ci)/2)
    auc_interval = (auc - z_score*auc_std, auc+z_score*auc_std)
    return auc_interval, auc

def cal_auc_ci(pred, y, bootstraps=100, fold_size=1000):
    classes = [0, 1]
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df  = pd.DataFrame(columns = ['y', 'pred'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence),replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1-prevalence)), replace = True)
            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values]
        )
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics

def cal_acc_ci(pre_binary, true_labels, ci=0.95):
    n = len(pre_binary)
    acc = accuracy_score(true_labels, pre_binary)
    acc_var = (acc * (1-acc)) / (n-1)
    acc_std = np.sqrt(acc_var)
    z_score = norm.ppf(1- (1-ci)/2)
    acc_interval = (acc - z_score*acc_std, acc+z_score*acc_std)
    return acc_interval, acc

def test_model(predictions, labels):
    # predicitions: positive_label

    pre_prob = predictions
    true_labels = labels
    CI = 0.95
    
    np.save('predictions.npy', predictions)
    print(predictions.shape)
    np.save('true_labels.npy', labels)

    pre_binary = [1 if p>= 0.5 else 0 for p in predictions]
    
    tn, fp, fn, tp = confusion_matrix(true_labels, pre_binary).ravel()
    print(f'{tn},{fp},{fn},{tp}')

    statistics = cal_auc_ci(pre_prob, true_labels)
    acc_ci, acc = cal_acc_ci(pre_binary, true_labels, CI)

    #print(f'AUC:{auc:.4f},{auc_ci[0]:.4f}-{auc_ci[1]:.4f}')
    print(f'ACC:{acc:.4f},{acc_ci[0]:.4f}-{acc_ci[1]:.4f}')
    print(f'mean:{",".join([f"{x:.4f}" for x in np.mean(statistics, axis=1)])}')
    print(f'max: {",".join([f"{x:.4f}" for x in np.max (statistics, axis=1)])}')
    print(f'min: {",".join([f"{x:.4f}" for x in np.min (statistics, axis=1)])}')
