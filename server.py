from collections import OrderedDict, defaultdict
from omegaconf import DictConfig
import torch
import wandb
def get_fit_metrics_aggregation_fn():
    def weighted_average(metrics):
        """
        Computes the weighted average of metrics across clients.

        Args:
        metrics: A list of tuples where each element represents the metrics from one client:
            - The first element in the tuple is the number of examples (int).
            - The second element is a dictionary of metrics where keys are metric names (str)
              and values are lists of floats representing metrics over epochs.

        Returns:
        A dictionary whose keys are all the metric names found among the input,
        and whose values are the weighted averages of the metrics (weighted by the number of examples).
        """
        # Initialize structures to store aggregated sums and weights
        total_examples = 0
        aggregated_metrics = {}

        # Iterate through each client's metrics
        for num_examples, client_metrics in metrics:
            total_examples += num_examples
            for metric_name, metric_values in client_metrics.items():

                # Use the last epoch value

                ## Training metrics are lists of floats
                if type(metric_values) == list:
                    last_value = max(metric_values)

                ## Validation metrics are floats
                else:
                    last_value = metric_values
                
                # Update the weighted sum for the metric
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = 0
                aggregated_metrics[metric_name] += num_examples * last_value

        # Compute the final weighted averages
        averaged_metrics = {
            metric_name: aggregated_sum / total_examples
            for metric_name, aggregated_sum in aggregated_metrics.items()
        }

        return averaged_metrics

    return weighted_average


def get_on_fit_config(cfg: DictConfig):

    def fit_config_fn(server_round: int):

        if server_round > 2:
            # cfg.client_config.lr = cfg.client_config.lr * 0.8
            #FIXME: Implement a learning rate scheduler for server rounds
            pass
        
        cfg.update({"current_round": server_round})
        return cfg

    return fit_config_fn

def get_evaluate_fn(model, dataloader, criterion, metrics):

    def evaluate_fn(server_round: int, parameters, config):
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        model.eval() 
        
        metrics_list = defaultdict(list)
        epoch_metrics = defaultdict(float)
        num_batches = 0
        with torch.no_grad():
            for x, y in dataloader["Test"]:
                inputs, targets = x.to(device), y.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Compute the metrics for the current batch
                batch_metrics = metrics(outputs, targets)
                batch_metrics.update(criterion.get_losses(outputs, targets))  # Add additional losses if any
                batch_metrics["loss"] = loss.item()

                # Accumulate batch metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                num_batches += 1


            # Average metrics over all batches
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= num_batches

            # Store epoch metrics for tracking
            for key, value in epoch_metrics.items():
                metrics_list[key].append(value)

            return loss.item(), dict(epoch_metrics)

    return evaluate_fn