import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np


def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            logits = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, thr_roc = roc_curve(list(gt), pred)  # 计算真正率和假正率
        # np.save('fpr.npy', fpr)
        # np.save('tpr.npy', tpr)
        auroc = auc(fpr, tpr)  # 计算auc的值

        precision, recall, thr_pr = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        # np.save('precision.npy', precision)
        # np.save('recall.npy', recall)

        viz.lines('ROC', tpr, fpr)
        viz.plot_lines('AUROC', auroc)
        # viz.lines('pred_scores', pred)
        viz.plot_lines('pr_auc', pr_auc)

        return auroc
