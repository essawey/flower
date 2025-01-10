from omegaconf import DictConfig
import torch
from collections import OrderedDict


def get_on_fit_config(cfg: DictConfig):

    def fit_config_fn(server_round: int):
        
        if server_round > 2:
            pass
            #FIXME: Implement a learning rate scheduler for server rounds

        return {
            'lr': cfg.client_config.lr,
            "local_epochs": cfg.client_config.local_epochs,
            "step_size": cfg.client_config.step_size,
            "gamma": cfg.client_config.gamma,
        }
    

    return fit_config_fn

def get_evaluate_fn(model, test_dataloader, criterion, metric):

    def evaluate_fn(server_round: int, parameters, config):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        
        # model_state = model.state_dict()
        # for k in model_state.keys():
        #     if model_state[k].shape != state_dict[k].shape:
        #         print(f"Shape mismatch : Loacl Model {model_state[k]}, Server Model {state_dict[k]}")
        #         print(f"In layer {k}")
        #         print()
                # '''
                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer inc.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer inc.double_conv.4.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down1.maxpool_conv.1.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down1.maxpool_conv.1.double_conv.4.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down2.maxpool_conv.1.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down2.maxpool_conv.1.double_conv.4.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down3.maxpool_conv.1.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down3.maxpool_conv.1.double_conv.4.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down4.maxpool_conv.1.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer down4.maxpool_conv.1.double_conv.4.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer up1.conv.double_conv.1.num_batches_tracked

                # Shape mismatch : Loacl Model 0, Server Model tensor([])
                # In layer up1.conv.double_conv.4.num_batches_tracked
                # '''

        for k, v in state_dict.items():
            # BatchNorm layers have a shape of torch.Size([0])
            # We need to convert them to torch.Size([1]) to load the model state dict
            if 'num_batches_tracked' in k and v.shape == torch.Size([0]):
                state_dict[k] = torch.tensor(0)

        model.load_state_dict(state_dict, strict=True)

        model.to(device)

        model.eval() 

        running_ious, running_losses, count = 0, 0, 0

        # Validation loop
        for x, y in test_dataloader:
            # Send to device (GPU or CPU)
            inputs = x.to(device)
            targets = y.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                
                # Calculate the loss
                loss = criterion(outputs, targets)
                loss_value = loss.item()
                running_losses += loss_value

                # Calculate the IoU
                iou = metric(outputs, targets)
                iou_value = iou.item()
                running_ious += iou_value
                count += 1

        val_loss = running_losses / count
        val_iou = running_ious / count

        print(
            f'''
            SERVER EVALUATION fn:
            Validation loss: {val_loss:.3f}
            Validation IoU: {val_iou:.3f}
            '''
        )

        return val_loss, {"Server Validation IoU" : val_iou}

    return evaluate_fn