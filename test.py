import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            logits = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)  ###计算真正率和假正率
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)  ###计算auc的值
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, pr_auc

