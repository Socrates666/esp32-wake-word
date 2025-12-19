import torch
import torch as nn

def IoULoss(feature, label):
    list = []
    for f, l in zip(feature, label):
        if ((torch.sigmoid(f).item())>0.5) != l[0].item():
            list.append(abs(f[0]-l[0]))
            continue
        f_up = f[2].item()
        f_dn = f[1].item()
        l_up = l[2].item()
        l_dn = l[1].item()
        if f_dn >= l_up:
            iou = 0.00
        elif f_dn >= l_dn:
            iou = (l_up-f_dn)/(f_up-l_dn)
        elif f_up < l_up:
            iou = (l_up - f_up)/(l_dn-f_dn)
        else:
            iou = 0
        list.append(torch.tensor(1-iou))
    return torch.tensor(list)

def FocalAndSmooth(feature, label):
    criterion = nn.BCEWithLogitsLoss()
    list = []
    for index, f, l in enumerate(feature, label):
        focal = criterion(feature[0], label[0])
        x1 = abs(f[2]-l[2])
        x2 = abs(f[1]-l[1])
        if x1 < 1:
            x1 = 0.5*x1*x1
        else:
            x1 -= 0.5
        if x2 < 1:
            x2 = 0.5*x2*x2
        else:
            x2 -= 0.5
        list.append(focal + x1 + x2)
    return torch.tensor(list)


