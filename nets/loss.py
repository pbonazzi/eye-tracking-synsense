"""
Implementation of Yolo Loss Function from the original yolo paper
"""
import pdb
import torch
import torch.nn as nn
import numpy as np
from nets.model import arguments

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """


    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


class GaussianLoss(nn.Module):
    def __init__(self, threshold):
        super(GaussianLoss, self).__init__()
        self.threshold = threshold # Mean of the Gaussian distribution

    def forward(self, y_pred):
        mu = self.threshold 
        sigma = 1.0  

        # Compute the probability density function (PDF) of the Gaussian distribution
        pdf = torch.exp(-0.5 * ((y_pred - mu) / sigma)**2) / (sigma * torch.sqrt(2 * torch.tensor(np.pi)))

        # Compute the loss as the distance from the Gaussian PDF
        loss = 1.0 - pdf

        return loss
    

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, args=arguments()):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (WiderFace is 1),
        """
        self.S = args.SxS_Grid
        print(f'Loss SxS_Grid {self.S}')     
        self.B = args.num_boxes
        self.C = args.num_classes

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.box_loss = 0
        self.object_loss = 0
        self.no_object_loss = 0
        self.class_loss = 0
        self.loss = 0
        self.args = args

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputed and have values between -1, 1 or -128, 127
        # if self.args.act_mode_8bit:
        #     predictions =  predictions.div(2**9.).add(1.)
            # predictions =  predictions.add(128.).div(2**8.)
        # predictions =  predictions.div(2**12.)
        # predictions =  predictions.div(100.)

        #pdb.set_trace()

        try:
            predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
            
            # Calculate IoU for the two predicted bounding boxes with target bbox
            iou_b1 = intersection_over_union(predictions[..., (self.C+1):(self.C+1+4)], target[..., (self.C+1):(self.C+1+4)])
            iou_b2 = intersection_over_union(predictions[..., (self.C+6):(self.C+10)], target[..., (self.C+1):(self.C+1+4)])
            ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

            # Take the box with highest IoU out of the two prediction
            # Note that bestbox will be indices of 0, 1 for which bbox was best
            iou_maxes, bestbox = torch.max(ious, dim=0)
            exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            # Set boxes with no object in them to 0. We only take out one of the two 
            # predictions, which is the one with highest Iou calculated previously.
            box_predictions = exists_box * ((bestbox * predictions[..., (self.C+6):(self.C+10)] + (1 - bestbox) * predictions[..., (self.C+1):(self.C+1+4)]))
            box_targets = exists_box * target[..., (self.C+1):(self.C+1+4)]
            target, prediction = box_targets.detach(), box_predictions.detach()
        
            # # Take sqrt of width, height of boxes to ensure that
            # box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
            # box_targets[..., 2:4] = torch.sign(box_targets[..., 2:4]) * torch.sqrt(  torch.abs(box_targets[..., 2:4] + 1e-6))
            box_loss = self.mse( torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2),)
            central_point = box_predictions[..., :2]+(box_predictions[..., 2:] - box_predictions[..., :2])/2

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            # pred_box is the confidence score for the bbox with highest IoU
            pred_box = ( bestbox * predictions[..., (self.C+5):(self.C+6)] + (1 - bestbox) * predictions[..., self.C:(self.C+1)] )
            object_loss = self.mse( torch.flatten(exists_box * pred_box),torch.flatten(exists_box * target[..., self.C:(self.C+1)]),)

            # no_object_loss = self.mse(
            #     torch.flatten((1 - exists_box) * predictions[..., self.C:(self.C+1)], start_dim=1),
            #     torch.flatten((1 - exists_box) * target[..., self.C:(self.C+1)], start_dim=1),
            # )

            # no_object_loss += self.mse(
            #     torch.flatten((1 - exists_box) * predictions[..., (self.C+5):(self.C+6)], start_dim=1),
            #     torch.flatten((1 - exists_box) * target[..., self.C:(self.C+1)], start_dim=1)
            # )


            # ======================= #
            #   FOR NO OBJECT LOSS    #
            # ======================= #

            loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
            )
            self.box_loss = box_loss
            self.object_loss = object_loss
            self.loss = loss

            loss_dict = {
                "box_loss" : box_loss,
                "object_loss" : object_loss,
                "loss" : loss
            }

            log_data = {
                "box_target":target,
                "box_pred":prediction,
                "central_point":central_point
            }
            
            return loss_dict, log_data
        
        except Exception as e:
            print(e)
            pdb.set_trace()