from omegaconf import DictConfig
import torch
from collections import OrderedDict

def get_fit_metrics_aggregation_fn():
    def weighted_average(metrics):
        """
        metrics: A list of tuples where:
        - The first element is the number of examples (int).
        - The second element is a dictionary of arbitrary metrics, e.g. {"train_loss": 0.2, "val_accuracy": 0.8, ...}

        Returns:
        A dictionary whose keys are all the metric names found among the input, and whose values
        are the weighted averages (weighted by the number of examples).
        """

        # 1. Collect all possible keys
        all_keys = set()
        for _, metric_dict in metrics:
            all_keys.update(metric_dict.keys())

        # 2. Initialize accumulators for each key
        weighted_sums = {key: 0.0 for key in all_keys}
        total_examples = 0

        # 3. Accumulate weighted sums
        for num_examples, metric_dict in metrics:
            total_examples += num_examples
            for key in all_keys:
                # If a key is missing for a particular client, decide how you want to handle it:
                # Option A: Treat missing as 0
                value = metric_dict.get(key, 0.0)
                # Option B: Skip if missing (would need more sophisticated logic)

                weighted_sums[key] += num_examples * float(value)

        # 4. Compute weighted averages
        if total_examples == 0:
            # Edge case: if total_examples is 0, return zeros or handle however appropriate
            return {key: 0.0 for key in all_keys}

        weighted_avgs = {
            key: weighted_sums[key] / total_examples for key in all_keys
        }

        return weighted_avgs

    return weighted_average


def get_on_fit_config(cfg: DictConfig):

    def fit_config_fn(server_round: int):
        
        if server_round > 2:
            pass
            #FIXME: Implement a learning rate scheduler for server rounds
        return {
            'lr': cfg.lr,
            "local_epochs": cfg.local_epochs,
            "step_size": cfg.step_size,
            "gamma": cfg.gamma,
        }
    

    return fit_config_fn

def get_evaluate_fn(model, dataloader, criterion, MeanDice, MeanIoU):

    def evaluate_fn(server_round: int, parameters, config):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        running_dices, running_ious, running_losses, count = 0, 0, 0, 0

        # Validation loop
        for x, y in dataloader["Test"]:
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
                # iou = MeanIoU(outputs, targets)
                # iou_value = iou.item()
                # running_ious += iou_value

                # dice = MeanDice(outputs, targets)
                # dice_value = dice.item()
                # running_dices += dice_value

                count += 1

        val_loss = running_losses / count
        val_iou = running_ious / count
        val_dice = running_dices / count

        print(
            f'''
            SERVER EVALUATION fn:
            Validation loss: {val_loss:.3f}
            Validation IoU: {val_iou:.3f}
            Validation Dice: {val_dice:.3f}
            '''
        )

        return val_loss, {"Server Validation IoU" : val_iou, "Server Validation Dice" : val_dice}

    return evaluate_fn