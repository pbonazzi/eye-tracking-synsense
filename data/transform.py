from typing import Tuple
import torch, pdb
from tonic.io import make_structured_array

class AedatEventsToXYTP:
    def __init__(self):
        pass

    def __call__(self, data):
        x = data['coords'][:, 0]
        y = data['coords'][:, 1]
        t = data['ts']
        p = data['polarity']
        return make_structured_array(x, y, t, p)

class FromPupilCenterToBoundingBox:
    def __init__(
        self, 
        image_size: Tuple[int, int] = (640, 480), 
        SxS_Grid: int=5,
        num_classes: int=1, 
        num_boxes: int=2,

    ):
        self.image_size = image_size
        self.delta = 10
        self.S, self.C, self.B = SxS_Grid, num_classes, num_boxes
    
    def __call__(self, target):
        x_norm, y_norm = target[0]/self.image_size[0],  target[1]/self.image_size[1]
        x_1, y_1 = x_norm-self.delta,y_norm-self.delta
        x_2, y_2 = x_norm+self.delta,y_norm+self.delta

        # grid
        label_matrix = torch.zeros((self.S, self.S, self.C +  5 * self.B), dtype=torch.float)
        row, column = int(self.S * y_norm), int(self.S * x_norm)
        
        # label
        label_matrix[row, column, self.C] = 1 # obj conf
        box_coordinates = torch.tensor([x_1, y_1, x_2, y_2])
        label_matrix[row, column, (self.C+1):(self.C+1+4)] = box_coordinates # box coord
        label_matrix[row, column, 0] = 1 # class

        return label_matrix

