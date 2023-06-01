import torch
import torch.nn as nn

class arguments():
    def __init__(self):
        self.device = torch.device("cuda")
        self.SxS_Grid = 5
        self.num_boxes = 2
        self.num_classes = 1
        self.input_channel = 1
        self.dsize = [128, 128]
        self.act_mode_8bit = None


class SynSenseEyeTracking(nn.Module):
    def __init__(self, args=arguments()):
        super(SynSenseEyeTracking, self).__init__()
        self.S, self.B, self.C = args.SxS_Grid, args.num_boxes, args.num_classes #num_classes-1
        self.name = "synsense"
        
        self.seq = nn.Sequential(
            # P1
            nn.Conv2d(in_channels=args.input_channel, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(4, 4),

            # P2
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            #nn.AvgPool2d(2, 2),

            # C2F 3
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=1, bias=True),
            nn.ReLU(),
            #nn.AvgPool2d(2, 2),

            # C2F 5
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            #nn.AvgPool2d(2, 2),

            # SPPF 6
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            #nn.AvgPool2d(2, 2),

            # SPPF 7
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(4, 4),

            # Dense Layer 8
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Dense Layer 9
            nn.Linear(128, self.S * self.S *(self.C + self.B * 5)),
            #nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)
