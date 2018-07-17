import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


class TransformedDataset(Dataset):
    
    def __init__(self, ds, xy_transform):
        self.ds = ds
        self.xy_transform = xy_transform
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.xy_transform(self.ds[index])
    
    
def logits_to_img(logits):
    assert logits.ndimension() == 3
    c, h, w = logits.shape
    assert c == 1
    logits = logits.detach().view(-1)
    y_probas = F.softmax(logits, dim=0)
    y_probas = y_probas.cpu()
    _, y_pred_cls = y_probas.max(dim=0)
    y_pred = torch.zeros((h * w), dtype=torch.uint8)
    y_pred[y_pred_cls] = 1
    return 255 * y_pred.numpy().reshape((h, w))


def logits_to_index(logits, width):
    assert logits.ndimension() == 3
    c, h, w = logits.shape
    assert c == 1
    logits = logits.detach().view(-1)
    y_probas = logits.cpu()
    _, y_pred_cls = y_probas.max(dim=0)
    y_pred_cls = y_pred_cls.item()
    return (y_pred_cls % width, y_pred_cls // width)


def normalize(x, size):
    return [(x[0] - 0.5 * size) / size, (x[1] - 0.5 * size) / size]
