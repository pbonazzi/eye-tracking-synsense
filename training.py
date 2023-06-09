import torch, pdb, fire, wandb, os, math, cv2, time
import numpy as np
from tqdm import tqdm

import tonic
from data.dataset import EyeTrackingInivationDataset
from data.transform import FromPupilCenterToBoundingBox, AedatEventsToXYTP, CenterCropEventsToSpeckResolution
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset, SlicedDataset, MemoryCachedDataset
from tonic.slicers import SliceByTime

from tonic.transforms import Compose, ToFrame

from nets import get_summary
from nets.model import SynSenseEyeTracking
from nets.loss import YoloLoss, GaussianLoss, arguments
from nets.lpf import LPFOnline
from util import draw_image

from sinabs.exodus import conversion
from sinabs.from_torch import from_model
from sinabs.exodus.layers import IAFSqueeze
from sinabs import SNNAnalyzer

def launch_fire(
    project_name = "synsense_snn", 
    arch_name = "BabyYoloV8", 
    dataset_name="EyeTrackingDataSet_FromInivation", 
    data_dir="data/example_data",
    output_dir="output/",
    img_width=128,
    img_height=128,
    batch_size=2,
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
    target_transforms = FromPupilCenterToBoundingBox()

    frame_transform =  ToFrame(
                sensor_size=[128, 128, 1],
                n_event_bins=100
            )
    input_transforms = Compose(
        [
            AedatEventsToXYTP(),
            CenterCropEventsToSpeckResolution(),
            frame_transform
        ]
    )
    target_transforms = FromPupilCenterToBoundingBox()
    
    # Dataset
    train_dataset = EyeTrackingInivationDataset(data_dir, 
                                                transform=input_transforms,
                                                target_transform=target_transforms,
                                                save_to="./data", 
                                                list_experiments=list(range(0, 26))) 
    
    test_dataset = EyeTrackingInivationDataset(data_dir, 
                                                transform=input_transforms,
                                                target_transform=target_transforms,
                                                save_to="./data", 
                                                list_experiments=list(range(26, 28))) 
    
    train_cache_dataset = DiskCachedDataset(
                        train_dataset, 
                        cache_path="/home/thor/projects/pietro/.cache/eye-tracking-tonic-dataset"
    )
    test_cache_dataset = DiskCachedDataset(
                        test_dataset, 
                        cache_path="/home/thor/projects/pietro/.cache/eye-tracking-tonic-dataset-test"
    )
    train_dataloader = DataLoader(train_cache_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True)

    test_dataloader = DataLoader(test_cache_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True)

    # pdb.set_trace()

    # train_dataset = SlicedDataset(
    #     train_dataset, 
    #     slicer=SliceByTime(time_window=50000), 
    #     transform=frame_transform, 
    #     metadata_path="./cache/slice-eye-tracking-tonic-dataset"
    # )

    # pdb.set_trace()

    
    # Model 
    sinabs_model = from_model(model.seq, 
                              add_spiking_output=False,  
                              synops=False,  
                              batch_size=batch_size)

    lpf = LPFOnline(num_channels=275, kernel_size=100).to(device)
    
    exodus_model = conversion.sinabs_to_exodus(sinabs_model).to(device)

    sinabs_analyzer = SNNAnalyzer(exodus_model.spiking_model)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    args = arguments()
    general_criterion = YoloLoss(args)
    spiking_criterion = GaussianLoss(threshold=1) # firing rate
    min_accuracy = float("inf")

    # Visualize and display training loss in a progress bar
    pbar = tqdm(range(num_epochs), desc= "Training")

    # backprop over epochs
    for epoch in pbar:
        # over batches
        model = model.train()
        for i, (data, labels) in enumerate(tqdm(train_dataloader)):
            # reset grad to zero for each batch
            optimizer.zero_grad()

            # port to device
            data, labels = data.float().to(device), labels.float().to(device)

            # forward pass
            b, t, c, h, w = data.shape
            data = data.reshape(b*t, c, h, w)
            outputs = exodus_model.spiking_model(data)

            # layer logging
            layer_stats = sinabs_analyzer.get_layer_statistics() 

            for key in layer_stats["parameter"].keys(): 
                wandb.log({f"{key}_{type(model.seq[int(key)]).__name__}/synops": layer_stats["parameter"][key]["synops"]})
                wandb.log({f"{key}_{type(model.seq[int(key)]).__name__}/synops_s": layer_stats["parameter"][key]["synops/s"]})


            for key in layer_stats["spiking"].keys(): 
                wandb.log({f"{key}_{type(model.seq[int(key)]).__name__}/firing_rate": layer_stats["spiking"][key]["firing_rate"]})

                # firing_rate_tensor = layer_stats["spiking"][key]["firing_rate_per_neuron"].cpu().detach().numpy()
                # if len(firing_rate_array.shape) == 1:
                #     continue
                # for k in range(firing_rate_array.shape[0]):
                #     image = wandb.Image(firing_rate_array[k], caption=f"Firing Rate Channel {k+1}")
                #     wandb.log({f"{key}_{type(model.seq[int(key)]).__name__}/channel_{k+1}": image})

            model_stats = sinabs_analyzer.get_model_statistics()
            for key in model_stats.keys():
                wandb.log({f"model_stats/{key}": model_stats[key].item()})


            # low pass filter
            outputs = outputs.reshape(b, outputs.shape[0]// b, outputs.shape[-1])
            outputs = outputs.permute(0, 2, 1)
            outputs = lpf(outputs)
            outputs = outputs[..., outputs.shape[-1]//2]
            
            # calculate loss
            loss_dict, log_data = general_criterion(outputs, labels)

            loss_dict["firing_rate"] = spiking_criterion(model_stats["firing_rate"])
            loss_dict["loss"] += loss_dict["firing_rate"]

            # display loss in progress bar
            pbar.set_postfix(loss=loss_dict["loss"].item())

            # backward pass
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            
            # optimize parameters
            optimizer.step()
            exodus_model.reset_states()

            # loggings
            target_box = labels[..., 2:6][0].detach()
            pred_box = log_data["box_pred"][0].detach()
            point_target = (target_box[..., :2] + (target_box[..., 2:] - target_box[..., :2])/2).sum(0).sum(0) 
            point_pred = (pred_box[..., :2] + (pred_box[..., 2:] - pred_box[..., :2])/2).sum(0).sum(0) 
            accuracy = math.sqrt(((point_pred[0]-point_target[0])*img_width)**2 + ((point_pred[1]-point_target[1])*img_height)**2)
            wandb.log({f"train/accuracy": accuracy})

            # save
            if min_accuracy<accuracy:
                accuracy = min_accuracy
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

        if epoch % 10 == 0:

            # With no gradient means less memory and calculation on forward pass
            with torch.no_grad():

                # evaluation usese Dropout and BatchNorm in inference mode
                model.eval()

                # Get the size of the first image
                width, height = 640, 480

                # Define the codec and create a VideoWriter object
                output_path = f"{epoch}.mp4"
                fps=5
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # over batches
                exec_time, accuracy = 0, 0
                for num, (imgs, labels) in enumerate(test_dataloader):

                    start = time.time()
                    # port to device
                    data, labels = data.float().to(device), labels.float().to(device)

                    # forward pass
                    b, t, c, h, w = data.shape
                    data = data.reshape(b*t, c, h, w)
                    outputs = model(data)

                    # low pass filter
                    outputs = outputs.reshape(b, outputs.shape[0]// b, outputs.shape[-1])
                    outputs = outputs.permute(0, 2, 1)
                    outputs = lpf(outputs)
                    outputs = outputs[..., outputs.shape[-1]//2]
                    end = time.time() 
                    exec_time += ((end-start)*1000)    

                    # calculate loss
                    loss_dict, log_data = general_criterion(outputs, labels)

                    # epoch images
                    target_box = labels[..., 2:6][0].detach()
                    pred_box = log_data["box_pred"][0].detach()
                    
                    # accuracy
                    point_target = (target_box[..., :2] + (target_box[..., 2:] - target_box[..., :2])/2).sum(0).sum(0) 
                    point_pred = (pred_box[..., :2] + (pred_box[..., 2:] - pred_box[..., :2])/2).sum(0).sum(0) 
                    accuracy += math.sqrt(((point_pred[0]-point_target[0])*args.dsize[0])**2 + ((point_pred[1]-point_target[1])*args.dsize[1])**2)
                    
                    log_img = draw_image(imgs[0], pred_box, target_box, point_pred, point_target)
                    log_img.rotate(90)
                    #wandb.log({"val/images": wandb.Image(log_img.rotate(90))})
                    cv_image = cv2.cvtColor(np.array(log_img), cv2.COLOR_RGB2BGR)
                    video_writer.write(cv_image) 


                video_writer.release()
                print("Video Released")


                # epoch loggings
                for key in loss_dict.keys():
                    wandb.log({f"val/{key}": loss_dict[key].detach()/batch_size})

                wandb.log({f"val/accuracy": round(accuracy/len(test_dataloader), 2)})
                print(f"Accuracy {round(accuracy/len(test_dataloader), 2)} (px)")
                print(f"Latency {round(exec_time/len(test_dataloader), 2)} (ms)")
           
    return model


if __name__ == '__main__':
  fire.Fire(launch_fire)

pdb.set_trace()
