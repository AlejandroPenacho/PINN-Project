import numpy as np
import torch
import matplotlib.pyplot as plt

import json
from typing import Optional, Type
import copy

class NN(torch.nn.Module):
    def __init__(self, layers, activation_function=torch.tanh):
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
        for i in range(self.n_layers-1):
            x = self.sigma(self.Ws[i] @ x + self.bs[i])
        
        return self.Ws[self.n_layers-1] @ x + self.bs[self.n_layers-1]
    
    

class CompleteConfig:
    def __init__(self, model_config, trainer_config, loss_blocks, function_dictionary):
        self.model_config: ModelConfig = model_config
        self.trainer_config: TrainerConfig = trainer_config
        self.loss_blocks: list[LossBlock] = loss_blocks
        self.function_dictionary = function_dictionary
        
    def to_dict(self):
        return {
            "model_config": self.model_config.to_dict(),
            "trainer_config": self.trainer_config.to_dict(),
            "loss_blocks": [x.to_dict() for x in self.loss_blocks]
        }
        
    @classmethod
    def from_dict(cls, info, function_dictionary):
        return cls(
            info["model_config"],
            info["trainer_config"],
            [LossBlock.from_dict(x) for x in info["loss_blocks"]],
            function_dictionary
        )
        
    def save_json(self, filename):
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def load_json(cls, filename, function_dictionary):
        with open(filename) as file:
            return cls.from_dict(json.load(file), function_dictionary)


    def train_model(self):
        stats: TrainingStats = TrainingStats(self)
        
        self.model_config.activation_function = self.function_dictionary[self.model_config.activation_function]
        self.trainer_config.optimizer_config.optimizer = self.function_dictionary[self.trainer_config.optimizer_config.optimizer]
        self.trainer_config.update_state = self.function_dictionary[self.trainer_config.update_state]
        for loss_block in self.loss_blocks:
            loss_block.loss_function = self.function_dictionary[loss_block.loss_function]
            loss_block.update_function = self.function_dictionary[loss_block.update_function]
        

        model = self.model_config.create_model()
        trainer_components = self.trainer_config.create_components(model)
        optimizer = trainer_components.optimizer
        

        self.trainer_config.update_state(self.trainer_config, stats, None)
        for loss_element in self.loss_blocks:
            loss_element.update(stats, None)

        
        for epoch in range(self.trainer_config.parameters["n_epochs"]):
            self.trainer_config.update_state(self.trainer_config, stats, epoch)
            for loss_element in self.loss_blocks:
                loss_element.update(stats, epoch)

            optimizer.zero_grad()
            total_loss = 0
            for loss_element in self.loss_blocks:
                individual_loss = torch.mean(loss_element.compute_local_loss(model))

                stats.individual_losses[loss_element.name].append(individual_loss.detach().numpy())
                total_loss += individual_loss
                # total_loss += loss_element.compute_total_loss(model)
                
            stats.total_loss_array.append(total_loss.detach().numpy())
                
            if epoch % 5000 == 0:
                print(total_loss)
                
            total_loss.backward()
            optimizer.step()
        
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
        self.update_state = update_function
        self.parameters = parameters
        self.state = {}


    def to_dict(self):
        return {
            "optimizer_config": self.optimizer_config.to_dict(),
            "lr_scheduler_config": self.lr_scheduler_config.to_dict() if self.lr_scheduler_config is not None else None,
            "update_state": self.update_state,
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, info):
        return cls(
            OptimizerConfig.from_dict(info["optimizer_config"]),
            LrSchedulerConfig.from_dict(info["lr_scheduler_config"]),
            "update_state",
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
        self.args = args
        
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


class LambdaSchedulerConfig:
    def __init__(self, lambda_scheduler, **args):
        self.lambda_scheduler = lambda_scheduler
        self.args = args

    def create_scheduler(self):
        return None

    def to_dict(self):
        return {"lambda_scheduler": self.lambda_scheduler, "args": self.args}

    @classmethod
    def from_dict(cls, info):
        if info is None:
            return None
        else:
            return cls(info["lambda_scheduler"], **info["args"])


class LossBlock:
    def __init__(
        self,
        name: str,
        loss_function,
        update_function,
        parameters
    ):
        self.name = name
        self.x = None
        self.y = None
        self.w = None
        self.parameters = parameters
        self.state = None

        self.loss_function = loss_function
        self.update_function = update_function
        

    def to_dict(self):
        return {
            "name": self.name,
            "loss_function": self.loss_function,
            "update_function": self.update_function,
            "parameters": self.parameters
        }
        
    @classmethod
    def from_dict(cls, info):
        return cls(info["name"], info["loss_function"], info["update_function"], info["parameters"])

        
    def update(self, epoch, stats):
        self.update_function(self, epoch, stats)
        
    def compute_local_loss(self, model):
        return self.loss_function(model, self.x, self.y, self.state)
    
    def compute_total_loss(self, model):
        local_loss = self.compute_local_loss(model)
        return torch.mean(local_loss * self.w)


class TrainingStats:
    def __init__(self, config: CompleteConfig):
        self.total_loss_array = []
        self.individual_losses = {x.name: [] for x in config.loss_blocks}
        pass



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
        