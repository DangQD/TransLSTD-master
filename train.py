import os
import random
import time

import torch
import numpy as np
import pandas as pd

from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler
from metrics import evaluate_hf
from models.mymodel import MyModel


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


def code_type(path):
    type_class = pd.read_csv(path)['class'].tolist()
    type_class = [code_type + 1 for code_type in type_class]
    type_class.insert(0, 0.0)
    type_class = torch.tensor(type_class).to(device)
    return type_class


if __name__ == '__main__':
    seed = 6666
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    task = 'h'  # 'm' or 'h'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Parameters sensitivity
    code_size = 64  # 16 32 64 96 128
    attention_size = 42
    visit_size = 64  # 16 32 64 96 128
    time_size = 32  # 16 32 64 96 128
    hidden_size = 64
    batch_size = 32
    epochs = 30

    # attention
    code_att = True
    visit_att = False

    # RNN variants
    rnn_select = 'LSTM'  # RNN / LSTM / GRU

    # ablation study
    type_aware = True
    time_aware = True
    context_aware = True

    # visualization
    visualization = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('../data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    code_type_class_path0 = '../data/kmeans/0-th_window333-' + dataset + '.csv'
    code_type_class_path1 = '../data/kmeans/1-th_window333-' + dataset + '.csv'
    code_type_class_path2 = '../data/kmeans/2-th_window333-' + dataset + '.csv'
    code_type_class = [code_type(code_type_class_path0), code_type(code_type_class_path1), code_type(code_type_class_path2)]

    code_num = len(code_adj)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    test2_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    task_conf = {
        'h': {
            'dropout': 0.4,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-3, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = MyModel(code_num=code_num, type_num=10, code_size=code_size, attention_size=attention_size,
                    visit_size=visit_size, time_size=time_size, hidden_size=hidden_size, batch_size=batch_size,
                    output_size=output_size, dropout_rate=dropout_rate, code_att=code_att, visit_att=visit_att,
                    rnn_select=rnn_select, type_aware=type_aware, time_aware=time_aware, context_aware=context_aware,
                    visualization=visualization).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    auc1_list = []
    auc2_list = []
    f11_list = []
    f12_list = []
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, y, visit_intervals = train_data[step]
            output = model(code_x, code_type_class, visit_lens, visit_intervals).squeeze()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, auc1, f1_score1 = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
        test_loss, auc2, f1_score2 = evaluate_fn(model, test_data, loss_fn, output_size, test2_historical)
        auc1_list.append(auc1)
        auc2_list.append(auc2)
        f11_list.append(f1_score1)
        f12_list.append(f1_score2)
        torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
    print("max auc on training dataset:", max(auc1_list))
    print("max auc on test dataset:", max(auc2_list))
    print("max F1 on training dataset:", max(f11_list))
    print("max F1 on test dataset:", max(f12_list))
