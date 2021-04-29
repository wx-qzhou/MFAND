import torch
from sklearn import metrics

def evaluate(preds, labels, pos_weight=1):
    labels = labels.squeeze().reshape(-1, 1).squeeze()
    # print(preds)
    zes = torch.Tensor(torch.zeros(labels.shape[0])).type(torch.LongTensor).cuda()
    ons = torch.Tensor(torch.ones(labels.shape[0])).type(torch.LongTensor).cuda()
    tp = int(((preds >= 0.1) & (labels == ons)).sum())
    fp = int(((preds >= 0.1) & (labels == zes)).sum())
    fn = int(((preds < 0.1) & (labels == ons)).sum())
    tn = int(((preds < 0.1) & (labels == zes)).sum())
    tp = int(pos_weight * tp)
    fn = int(pos_weight * fn)
    
    epsilon = 1e-7
    acc = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return acc, recall, (2 * acc * recall) / (acc + recall + 1e-13), torch.tensor([tp, fp, fn, tn])

def evaluate_ma_mi(preds, labels, pos_weight=1):
    labels = labels.squeeze().reshape(-1, 1).squeeze()
    neg_weight = 1
    # print(preds)
    zes = torch.Tensor(torch.zeros(labels.shape[0])).type(torch.LongTensor).cuda()
    ons = torch.Tensor(torch.ones(labels.shape[0])).type(torch.LongTensor).cuda()
    tp = int(((preds >= 0.1) & (labels == ons)).sum())
    fp = int(((preds >= 0.1) & (labels == zes)).sum())
    fn = int(((preds < 0.1) & (labels == ons)).sum())
    tn = int(((preds < 0.1) & (labels == zes)).sum())
    tp0 = tn
    fp0 = fn
    fn0 = fp
    tn0 = tp

    tp = int(pos_weight * tp)
    fn = int(pos_weight * fn)
    tp0 = int(neg_weight * tp0)
    fn0 = int(neg_weight * fn0)

    epsilon = 1e-7
    acc = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acc0 = tp0 / (tp0 + fp0 + epsilon)
    recall0 = tp0 / (tp0 + fn0 + epsilon)
    f11 = (2 * acc * recall) / (acc + recall + 1e-13)
    f10 = (2 * acc0 * recall0) / (acc0 + recall0 + 1e-13)
    p_ma = (acc + acc0) / 2
    r_ma = (recall0 + recall) / 2
    f1_ma = 2 * p_ma * r_ma / (p_ma + r_ma)
    p_mi = (tp + tp0) / (tp + tp0 + fp + fp0)
    r_mi = (tp + tp0) / (tp + tp0 + fn + fn0)
    f1_mi = 2 * p_mi * r_mi / (p_mi + r_mi)
    return acc, recall, f11, torch.tensor([tp, fp, fn, tn]), torch.tensor([p_ma, p_mi, r_ma, r_mi, f1_ma, f1_mi])

def Mi_Macro_evaluate(preds, labels):
    preds = preds.cpu()
    labels = labels.type(torch.LongTensor)
    p_ma = metrics.precision_score(labels, preds, labels=[0,1], average='macro')
    p_mi = metrics.precision_score(labels, preds, labels=[0,1], average='micro')
    p_we = metrics.precision_score(labels, preds, labels=[0,1], average='weighted')
    
    r_ma = metrics.recall_score(labels, preds, labels=[0,1], average='macro')
    r_mi = metrics.recall_score(labels, preds, labels=[0,1], average='micro')
    r_we = metrics.recall_score(labels, preds, labels=[0,1], average='weighted')
    
    f1_ma = metrics.f1_score(labels, preds, labels=[0,1], average='macro')
    f1_mi = metrics.f1_score(labels, preds, labels=[0,1], average='micro')
    f1_we = metrics.f1_score(labels, preds, labels=[0,1], average='weighted')
    return torch.tensor([p_ma, p_mi, p_we, r_ma, r_mi, r_we, f1_ma, f1_mi, f1_we])

def cuda_list_object_1d(list_1d, device=None, non_blocking=False):
    return [e.cuda(device, non_blocking) for e in list_1d]
