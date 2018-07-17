import torch
from torch.utils.data import Dataset


class NotSoClevr(Dataset):
    
    def __init__(self, canvas_size=64, square_size=9):
        self.canvas_size = canvas_size
        self.square_size = square_size

        self.centers = [
            (i, j) for i in range(self.square_size//2, self.canvas_size - self.square_size//2 + 1) 
            for j in range(self.square_size//2, self.canvas_size - self.square_size//2 + 1)
        ]
            
    def __len__(self):
        return len(self.centers)
    
    def __getitem__(self, index):
        c = self.centers[index]
        p = torch.zeros((1, self.canvas_size, self.canvas_size), dtype=torch.uint8)
        p[0, c[1], c[0]] = 1
        im = torch.zeros((1, self.canvas_size, self.canvas_size), dtype=torch.uint8)
        im[
            0,
            c[1] - self.square_size//2:c[1] + self.square_size//2, 
            c[0] - self.square_size//2:c[0] + self.square_size//2
        ] = 1        
        return c, p, im
