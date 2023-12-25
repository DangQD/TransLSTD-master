import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd


def code_type(path):
    type_class = pd.read_csv(path)['class'].tolist()
    type_class = [code_type + 1 for code_type in type_class]
    type_class.insert(0, 0.0)
    type_class = torch.tensor(type_class).cuda()
    return type_class


dataset = 'mimic4'
code_type_class_path0 = '../data/kmeans/0-th_window333-' + dataset + '.csv'
code_type_class_path1 = '../data/kmeans/1-th_window333-' + dataset + '.csv'
code_type_class_path2 = '../data/kmeans/2-th_window333-' + dataset + '.csv'
code_type_class = [code_type(code_type_class_path0), code_type(code_type_class_path1), code_type(code_type_class_path2)]


def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=0)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks),))
    r = np.zeros((len(ks),))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / len(t)
    return a / len(y_true_hot), r / len(y_true_hot)


def evaluate_hf(model, dataset, loss_fn, output_size=1, historical=None):
    model.eval()
    total_loss = 0.0
    labels = dataset.label()
    outputs = []
    preds = []

    for step in range(len(dataset)):
        code_x, visit_lens, y, visit_intervals = dataset[step]
        output = model(code_x, code_type_class, visit_lens, visit_intervals).squeeze()
        loss = loss_fn(output, y)
        total_loss += loss.item() * output_size * len(code_x)
        output = output.detach().cpu().numpy()
        outputs.append(output)
        pred = (output > 0.5).astype(int)
        preds.append(pred)
        print('\r    Evaluating step %d / %d' % (step + 1, len(dataset)), end='')
    avg_loss = total_loss / dataset.size()
    outputs = np.concatenate(outputs)
    preds = np.concatenate(preds)

    auc = roc_auc_score(labels, outputs)
    f1_score_ = f1_score(labels, preds)
    print('\r    Evaluation: loss: %.4f --- auc: %.4f --- f1_score: %.4f' % (avg_loss, auc, f1_score_))
    return avg_loss, auc, f1_score_
