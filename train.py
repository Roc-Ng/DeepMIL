import torch
import torch.nn.functional as F


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2 - arr)**2)
    return lamda1 * loss


def sparsity(arr, lamda2):
    loss = torch.sum(arr)
    return lamda2 * loss


def ranking(scores, batch_size):
    loss = torch.tensor(0.0, requires_grad=True)
    for i in range(batch_size):
        maxn = torch.max(scores[int(i * 32):int((i + 1) * 32)])
        maxa = torch.max(scores[int(i * 32 + batch_size * 32):int((i + 1) * 32 + batch_size * 32)])
        tmp = F.relu(1.0 - maxa + maxn)
        loss = loss + tmp
        loss = loss + smooth(
            scores[int(i * 32 + batch_size * 32):int((i + 1) * 32 + batch_size * 32)],
            8e-5,
        )
        loss = loss + sparsity(
            scores[int(i * 32 + batch_size * 32):int((i + 1) * 32 + batch_size * 32)],
            8e-5,
        )
    return loss / batch_size


def train(nloader, aloader, model, batch_size, optimizer, viz, plot_freq, device):
    with torch.set_grad_enabled(True):
        model.train()
        n_iter = iter(nloader)
        a_iter = iter(aloader)
        for i in range(int(800 / batch_size)):
            ninput = next(n_iter)
            ainput = next(a_iter)
            ninput = ninput.view(batch_size * 32, -1)
            ainput = ainput.view(batch_size * 32, -1)
            input = torch.cat((ninput, ainput), 0).to(device)

            scores = model(input)  # b*32  x 2048
            loss = (ranking(scores, batch_size) + sparsity(scores, 8e-5) + smooth(scores, 8e-5))

            if i % plot_freq == 0:
                viz.plot_lines('loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
