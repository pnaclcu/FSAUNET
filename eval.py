import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import dice_coeff
def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_mask = 0
    step=0
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)
        step+=1
        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot_mask += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred1 = torch.sigmoid(mask_pred)
            pred1 = (pred1 > 0.5).float()
            tot_mask += dice_coeff(pred1, true_masks).item()
    net.train()
    return tot_mask / n_val
