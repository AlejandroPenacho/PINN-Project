import numpy as np
import torch
import matplotlib.pyplot as plt

import json
from typing import Optional, Type
import copy

class ModifiedMLP(torch.nn.Module):
    """A modified Neural Network for PINN proposed in some paper.
    
        This is not being used at the moment, so you can ignore it.
    """
    def __init__(self, layer_width: int, n_layers: int, input_size: int, output_size: int, activation_function=torch.tanh):
        self.n_layers = n_layers
        self.sigma = activation_function

        std_dev = torch.sqrt(2/(input_size + layer_width))
        self.W_u = torch.nn.Parameter(
            torch.normal(0, std_dev,
            (layer_width, input_size),
            requires_grad=True
        ))
        self.W_v = torch.nn.Parameter(
            torch.normal(0, std_dev,
            (layer_width, input_size),
            requires_grad=True
        ))
        self.b_u = torch.nn.Parameter(
            torch.normal(0, std_dev,
            (layer_width, 1),
            requires_grad=True
        ))
        self.b_v = torch.nn.Parameter(
            torch.normal(0, std_dev,
            (layer_width, 1),
            requires_grad=True
        ))
        
        self.Ws = torch.nn.ParameterList()
        self.bs = torch.nn.ParameterList()
        
        self.Ws.append(torch.normal(0, std_dev, (layer_width, input_size), requires_grad=True))
        self.bs.append(torch.normal(0, std_dev, (layer_width, 1), requires_grad=True))
       
        for _ in range(1, n_layers-1):
            std_dev = torch.sqrt(1/layer_width)
            self.Ws.append(torch.normal(0, std_dev, (layer_width, layer_width), requires_grad=True))
            self.bs.append(torch.normal(0, std_dev, (layer_width, 1), requires_grad=True))

        std_dev = torch.sqrt(2/(layer_width + output_size))
        self.Ws.append(torch.normal(0, std_dev, (output_size, layer_width), requires_grad=True))
        self.bs.append(torch.normal(0, std_dev, (output_size, 1), requires_grad=True))
        
    def forward(self, x):
        u = self.sigma(self.W_u @ x + self.b_u)
        v = self.sigma(self.W_v @ x + self.b_v)
        
        h = x

        for i in range(self.n_layers-1):
            z = self.sigma(self.Ws[i] @ h + self.bs[i])
            h = (1-z) * u + z * v

        return self.Ws[-1] @ h + self.bs[-1]


class NN(torch.nn.Module):
    """
        Basic Neural Network
    """
    def __init__(self, layers, activation_function=torch.tanh):
        """
            Initializes the neural network with the layers and activation fuction
            selected.

            # Arguments
            - layers: is given as an array of integers which correspond to the width of
                each layer.
            
            - activation_function: a function that goes from R to R.
        """
        super().__init__()
        self.Ws = torch.nn.ParameterList()
        self.bs = torch.nn.ParameterList()
        self.n_layers = len(layers)-1
        self.sigma = activation_function
        
        for i in range(self.n_layers):
            in_size = layers[i]
            out_size = layers[i+1]
            std_dev = np.sqrt(2/(in_size + out_size))
            W = torch.normal(
                0, std_dev,
                (out_size, in_size),
                requires_grad=True
            )
            b = torch.normal(
                0,
                std_dev,
                (out_size, 1),
                requires_grad=True
            )
            
            self.Ws.append(W)
            self.bs.append(b)
    
    def forward(self, x):
        """Compute the output of the nueral network for an input x"""

        for i in range(self.n_layers-1):
            x = self.sigma(self.Ws[i] @ x + self.bs[i])
        
        return self.Ws[self.n_layers-1] @ x + self.bs[self.n_layers-1]
    


class NNWithG(torch.nn.Module):
    """
        Basic Neural Network
    """
    def __init__(self, layers, layers_G, activation_function=torch.tanh):
        """
            Initializes the neural network with the layers and activation fuction
            selected.

            # Arguments
            - layers: is given as an array of integers which correspond to the width of
                each layer.
            
            - activation_function: a function that goes from R to R.
        """
        super().__init__()
        self.Ws = torch.nn.ParameterList()
        self.bs = torch.nn.ParameterList()
        self.Ws_G = torch.nn.ParameterList()
        self.bs_G = torch.nn.ParameterList()

        self.n_layers = len(layers)-1
        self.sigma = activation_function

        self.n_layers_G = len(layers_G)-1
        
        for i in range(self.n_layers):
            in_size = layers[i]
            out_size = layers[i+1]
            std_dev = np.sqrt(2/(in_size + out_size))
            W = torch.normal(
                0, std_dev,
                (out_size, in_size),
                requires_grad=True
            )
            b = torch.normal(
                0,
                std_dev,
                (out_size, 1),
                requires_grad=True
            )
            
            self.Ws.append(W)
            self.bs.append(b)
            
            
        for i in range(self.n_layers_G):
            in_size = layers_G[i]
            out_size = layers_G[i+1]
            std_dev = np.sqrt(2/(in_size + out_size))
            W = torch.normal(
                0, std_dev,
                (out_size, in_size),
                requires_grad=True
            )
            b = torch.normal(
                0,
                std_dev,
                (out_size, 1),
                requires_grad=True
            )
            
            self.Ws_G.append(W)
            self.bs_G.append(b)

    
    def forward(self, x):
        """Compute the output of the nueral network for an input x"""
        for i in range(self.n_layers-1):
            x = self.sigma(self.Ws[i] @ x + self.bs[i])
        
        return self.Ws[self.n_layers-1] @ x + self.bs[self.n_layers-1]


    def compute_g(self, x):
        """Compute the output of the nueral network for an input x"""
        for i in range(self.n_layers_G-1):
            x = self.sigma(self.Ws_G[i] @ x + self.bs_G[i])
        
        x = self.Ws_G[self.n_layers_G-1] @ x + self.bs_G[self.n_layers_G-1]
        return x
    

class CompleteConfig:
    """Complete set of parameters to create and train a neural network
    
    A complete configuration can be copied, stored and loaded from a json, and used
    to create and train a neural network. It is initialized with a model configuration,
    a trainer configuration and a dictionary of loss blocks.
    """

    def __init__(self, model_config, trainer_config, loss_blocks, function_dictionary):
        """Initializes the complete configuration
        
        # Arguments:
        - model_config: ModelConfig
            The configuration used to initialize the neural network to be trained.
            
        - trainer_config: TrainerConfig
            The configuration of the trainer that determines how to update the neural network
            base on the losses.
            
        - loss_blocks: dict[str, LossBlock]
            Each of the loss computation elements used to obtain the total loss of the NN.
            It is given as a dict, so each LossBlock is associated to a name that makes it easier
            to identify.
        
        - function_dictionary:
            A dictionary that takes a string and returns a python object. Since python objects can not
            be stored in json, CompleteConfig stores them as strings. In order to call them, they must
            be translated to the corresponding object. The dictionary relates each string to its object.
            For more information check "replace_function_names"

        """
        self.model_config: ModelConfig = model_config
        self.trainer_config: TrainerConfig = trainer_config
        self.loss_blocks: dict[str, LossBlock] = loss_blocks
        self.function_dictionary = function_dictionary
        

    def to_dict(self):
        """Produces a dictionary with all the informatin required to rebuild this object"""

        return {
            "model_config": self.model_config.to_dict(),
            "trainer_config": self.trainer_config.to_dict(),
            "loss_blocks": {k: v.to_dict() for k,v in self.loss_blocks.items()}
        }
        

    @classmethod
    def from_dict(cls, info, function_dictionary):
        """Initializes a complete configuration from a dictionary"""
        return cls(
            ModelConfig.from_dict(info["model_config"]),
            TrainerConfig.from_dict(info["trainer_config"]),
            {k: LossBlock.from_dict(v) for k, v in info["loss_blocks"].items()},
            function_dictionary
        )
        

    def save_json(self, filename):
        """Save the configuration as a json, so it can be stored and loaded later"""
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)


    @classmethod
    def load_json(cls, filename, function_dictionary):
        """Initialize the configuration from a json file"""

        with open(filename) as file:
            return cls.from_dict(json.load(file), function_dictionary)
        

    def replace_function_names(self):
        """Replaces all strings for its corresponding Python objects"""
        self.model_config.activation_function = self.function_dictionary[self.model_config.activation_function]
        self.trainer_config.optimizer_config.optimizer = self.function_dictionary[self.trainer_config.optimizer_config.optimizer]
        self.trainer_config.update_function = self.function_dictionary[self.trainer_config.update_function]

        if self.trainer_config.lr_scheduler_config is not None:
           self.trainer_config.lr_scheduler_config.lr_scheduler = self.function_dictionary[self.trainer_config.lr_scheduler_config.lr_scheduler]

        for loss_block in self.loss_blocks.values():
            loss_block.loss_function = self.function_dictionary[loss_block.loss_function]
            loss_block.update_function = self.function_dictionary[loss_block.update_function]


    def train_model(self, override_model=None):
        """
            Call this function to train the model. Training makes changes in the configuration
            since parameters can change with the epochs. To keep the configuration the same,
            we clone it and use the cloned one to do the training, so the original configuration
            remains the same.
        """
        copied_self = copy.deepcopy(self)
        model, stats = copied_self._actually_train_model(override_model)
        return (model, stats, copied_self)


    def _actually_train_model(self, override_model=None):
        """
            This is where the neural network is actually trained. I have tried to comment it.
        """

        # First initialize the stats, which are used to collect information of the training
        # like the evolution of the loss function
        stats: TrainingStats = TrainingStats(self)
        
        # Use the function dictionary to change the strings in the configuration by the actual
        # python functions.
        self.replace_function_names()

        # Initialize the model and the optimizer
        if override_model is None:
            model = self.model_config.create_model()
        else:
            model = override_model

        trainer_components = self.trainer_config.create_components(model)
        optimizer = trainer_components.optimizer
        lr_scheduler = trainer_components.lr_scheduler
        

        for epoch in range(self.trainer_config.parameters["n_epochs"]):
            # At each epoch, we first update the training parameters and the loss blocks.
            self.trainer_config.update_function(self.trainer_config, stats, epoch)
            for loss_element in self.loss_blocks.values():
                loss_element.update(stats, epoch)

            # Zero grad to restart the gradient. Otherwise bad stuff happens
            optimizer.zero_grad()
            total_loss = torch.tensor(0, dtype=torch.float)

            # For each loss block, we compute the loss of the model and add it to the total loss
            # Also, we save results to the stats.
            for loss_name, loss_element in self.loss_blocks.items():
                indiv_losses = loss_element.compute_loss(model)
                block_loss = torch.mean(indiv_losses)

                stats.last_individual_loss[loss_name] = indiv_losses.detach()
                stats.loss_per_block[loss_name][epoch] = block_loss.detach()

                total_loss += block_loss
                # total_loss += loss_element.compute_total_loss(model)
                
            stats.total_loss_array[epoch] = total_loss.detach()
                
            if epoch % 5000 == 0:
                print(total_loss)
                
            total_loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
        return model, stats
    
    

class ModelConfig:
    def __init__(self, layers: list[int], activation_function: str):
        self.layers = layers
        self.activation_function = activation_function
        
    def create_model(self) -> NN:
        return NN(self.layers, self.activation_function)
    
    def to_dict(self):
        return {"layers": self.layers, "activation_function": self.activation_function}
    
    @classmethod
    def from_dict(cls, info: str):
        return cls(info["layers"], info["activation_function"])


class TrainerConfig:
    def __init__(
        self,
        optimizer_config: Type["OptimizerConfig"],
        lr_scheduler_config: Optional[Type["LrSchedulerConfig"]],
        update_function,
        parameters
    ):

        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.update_function = update_function
        self.parameters = parameters

    def to_dict(self):
        return {
            "optimizer_config": self.optimizer_config.to_dict(),
            "lr_scheduler_config": self.lr_scheduler_config.to_dict() if self.lr_scheduler_config is not None else None,
            "update_function": self.update_function,
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, info):
        return cls(
            OptimizerConfig.from_dict(info["optimizer_config"]),
            LrSchedulerConfig.from_dict(info["lr_scheduler_config"]),
            info["update_function"],
            info["parameters"]
        )
        

    def create_components(self, model):
        optimizer = self.optimizer_config.create_optimizer(model)
        if self.lr_scheduler_config is not None:
            lr_scheduler = self.lr_scheduler_config.create_scheduler(optimizer)
        else:
            lr_scheduler = None
        
        return TrainerComponents(optimizer, lr_scheduler)


class TrainerComponents:
    def __init__(self, optimizer, lr_scheduler):
        self.optimizer: torch.optim.Optimizer = optimizer
        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = lr_scheduler


class OptimizerConfig:
    def __init__(self, optimizer, **args):
        self.optimizer = optimizer
        self.args: dict = args
        
    def create_optimizer(self, model: torch.nn.Module):
        return self.optimizer(model.parameters(), **self.args)
    
    def to_dict(self):
        return {"optimizer": self.optimizer, "args": self.args}

    @classmethod
    def from_dict(cls, info):
        return cls(info["optimizer"], **info["args"])


class LrSchedulerConfig:
    def __init__(self, lr_scheduler, **args):
        self.lr_scheduler = lr_scheduler
        self.args = args
        
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return self.lr_scheduler(optimizer, **self.args)

    def to_dict(self):
        return {"lr_scheduler": self.lr_scheduler, "args": self.args}

    @classmethod
    def from_dict(cls, info):
        if info is None:
            return None
        else:
            return cls(info["lr_scheduler"], **info["args"])


class LossBlock:
    def __init__(
        self,
        loss_function,
        update_function,
        parameters
    ):
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        self.w: Optional[torch.Tensor] = None
        self.parameters = parameters

        self.loss_function = loss_function
        self.update_function = update_function
        

    def to_dict(self):
        return {
            "loss_function": self.loss_function,
            "update_function": self.update_function,
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, info):
        return cls(info["loss_function"], info["update_function"], info["parameters"])

        
    def update(self, epoch, stats):
        self.update_function(self, epoch, stats)
        
    def compute_loss(self, model):
        return self.loss_function(model, self.x, self.y, self.parameters["loss_function_parameters"])


class TrainingStats:
    def __init__(self, config: CompleteConfig):
        self.total_loss_array = torch.zeros(config.trainer_config.parameters["n_epochs"])
        self.loss_per_block = {
            name: torch.zeros(config.trainer_config.parameters["n_epochs"])
            for name in config.loss_blocks.keys()
        }
        self.last_individual_loss = {
            name: None for name in config.loss_blocks.keys()
        }
    
    def save_results(self, epoch, data):
        self.total_loss_array.append(data["total_loss"])
        for k, v in data["loss_per_block"].items():
            self.loss_per_block[k].append(v)



def sine_check():
    x_arr = np.linspace(0, 5, 100)
    y_arr = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.sin(X + Y)

    x_train = np.stack(
        [X.reshape(-1), Y.reshape(-1)]
    )
    y_train = Z.reshape((1, -1))

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    
    
    nn = NN([2, 10, 10, 10, 10, 1])

    lr = 0.001
    optimizer = torch.optim.Adam(nn.parameters(), lr=lr)
    vanilla = False

    for i in range(6000):
        loss = torch.mean(torch.square(nn.forward(x_train) - y_train))

        if i % 1000 == 0:
            print(loss)
        loss.backward()

        if vanilla:
            for parameter in nn.parameters():
                with torch.no_grad():
                    parameter -= lr * parameter.grad
                parameter.grad.zero_()
        else:
            optimizer.step()
            optimizer.zero_grad()


    with torch.no_grad():
        z_show = nn.forward(x_train)
        z_show = z_show.reshape(100, 100).numpy()
    
    fig = plt.figure()
    axes = fig.subplots(1, 2)
    axes[0].pcolormesh(X, Y, Z)
    axes[1].pcolormesh(X, Y, z_show)
    axes[0].set_title("Reference")
    axes[1].set_title("Neural Network")
    
    plt.show()

        

if __name__ == "__main__":
    sine_check()
        