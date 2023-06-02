import torch, pdb, fire, wandb, os, math
from tqdm import tqdm

import tonic
from data.dataset import EyeTrackingInivationDataset
from data.transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP
from torch.utils.data import DataLoader
from tonic import MemoryCachedDataset
from tonic.transforms import Compose, ToFrame, CenterCropEventsToSpeckResolution

from nets import get_summary
from nets.model import SynSenseEyeTracking
from nets.loss import YoloLoss

from sinabs.exodus import conversion
from sinabs.from_torch import from_model
from sinabs.exodus.layers import IAFSqueeze

def launch_fire(
    project_name = "synsense_snn", 
    arch_name = "BabyYoloV8", 
    dataset_name="EyeTrackingDataSet_FromInivation", 
    data_dir="data/example_data",
    output_dir="output/",
    img_width=128,
    img_height=128,
    batch_size=64,
    num_epochs=100, 
    lr=1e-3
    
    ):

    wandb.init(
        project=project_name,
        dir= output_dir,
        config={
        "learning_rate": lr,
        "architecture": arch_name,
        "dataset": dataset_name,
        "epochs": num_epochs,
        }
    )

    out_dir = os.path.join(output_dir, "wandb", wandb.run.name)
    os.makedirs(out_dir, exist_ok=True)

    # Setting up environment
    device = torch.device("cuda")
    model = SynSenseEyeTracking()
    get_summary(model)
    
    # Transforms
    input_transforms = Compose(
        [
            AedatEventsToXYTP(),
            CenterCropEventsToSpeckResolution(),
            ToFrame(
                sensor_size=[128, 128, 1],
                n_event_bins=100
            )
        ]
    )
    target_transforms = FromPupilCenterToBoundingBox()

    # Dataset
    train_dataset = EyeTrackingInivationDataset(data_dir, 
                                                transform=input_transforms,
                                                target_transform=target_transforms,
                                                save_to="./data", 
                                                list_experiments=[0]) 
    augmented_dataset = MemoryCachedDataset(train_dataset)
    train_dataloader = DataLoader(augmented_dataset, 
                                  collate_fn=tonic.collation.PadTensors(batch_first=True),
                                  batch_size=batch_size, 
                                  shuffle=True)
    
    # Model 
    sinabs_model = from_model(model.seq, 
                              add_spiking_output=True,  
                              synops=False,  
                              batch_size=batch_size)
    
    exodus_model = conversion.sinabs_to_exodus(sinabs_model)

    #pdb.set_trace()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = YoloLoss()

    # Visualize and display training loss in a progress bar
    pbar = tqdm(range(num_epochs), desc= "Training")

    # backprop over epochs
    for epoch in pbar:
        # over batches
        model = model.train()
        for (data, labels) in enumerate(train_dataloader):
            # reset grad to zero for each batch
            optimizer.zero_grad()

            # port to device
            data, labels = data.to(device), labels.to(device)

            # forward pass
            outputs = exodus_model.spiking_model(data)
            
            # calculate loss
            loss_dict, log_data = criterion(outputs, labels)

            # display loss in progress bar
            pbar.set_postfix(loss=loss_dict["loss"].item())

            # backward pass
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            
            # optimize parameters
            optimizer.step()
        

        if epoch==0:
            torch.save(model.state_dict(), os.path.join(out_dir, "network.pt"))

        # loggings
        target_box = labels[..., 2:6][0].detach()
        pred_box = log_data["box_pred"][0].detach()
        point_target = (target_box[..., :2] + (target_box[..., 2:] - target_box[..., :2])/2).sum(0).sum(0) 
        point_pred = (pred_box[..., :2] + (pred_box[..., 2:] - pred_box[..., :2])/2).sum(0).sum(0) 
        accuracy = math.sqrt(((point_pred[0]-point_target[0])*img_width)**2 + ((point_pred[1]-point_target[1])*img_height)**2)
        wandb.log({f"train/accuracy": accuracy})
            
    return model


if __name__ == '__main__':
  fire.Fire(launch_fire)

pdb.set_trace()
