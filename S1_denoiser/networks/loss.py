import torch
import numpy as np

class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


class XEDiceLoss(torch.nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(self):
        super().__init__()
        self.xe = torch.nn.MSELoss()
        self.edge = Sobel()

    def forward(self, pred, true):
        #valid_pixel_mask = true.ne(255)  # valid pixel mask

        # Cross-entropy loss
        #temp_true = torch.where((true == 255), 0, true)  # cast 255 to 0 temporarily
        #print(f'pred max is {pred.max()}, true max is {true.max()}')
        maps = self.edge(true)
        ed_loss = self.xe(pred*maps, true*maps)
        xe_loss = self.xe(pred, true)
        #xe_loss = xe_loss.mean()

        # Dice loss
        # pred = torch.softmax(pred, dim=1)[:, 1]
        # pred = pred.masked_select(valid_pixel_mask)
        # true = true.masked_select(valid_pixel_mask)
        # dice_loss = 1 - (2.0 * torch.sum(pred * true)) / (torch.sum(pred + true) + 1e-7)

        return xe_loss+0.5*ed_loss


def intersection_and_union(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torc.Tensor): a tensor of labels

    Returns:
        intersection (int): total intersection of pixels
        union (int): total union of pixels
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum(), union.sum()
