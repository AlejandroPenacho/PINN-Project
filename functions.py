import torch
import neural
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Main Ideas

"""
The configuration for trainig a neural network is given by a neural.CompleteConfig. This configuration is composed by 3 elements:

- model_config, an instace of nn.ModelConfig, which specifies how a neural network should be first constructed. At the moment,
    this is used to set the layers and the activation function.

- trainer_config, an instance of nn.TrainerConfig, that is used to set the optimizer, the learning rate scheduler, the parameters
    of the training (for example, number of epochs or relative weight of each loss function) and a setup function that can modify
    these parameters at each epoch.

- loss_blocks, a dictionary of neural.LossBlock. A loss block contains all the data necessary to compute one specific loss of the 
    model. Its elements are: x, y and w, which are the test points, expected outputs and weight of each point. Also, a dictionary
    of parameters and a setup function that can modify the loss block at each epoch.
    

In order to define a complete configuration, it is necessary to define these 3 elements.

However, first we need the functions used in the training, which can only be kept as python functions.
"""


# Define functions

"""
Although we can store a lot of stuff as json files, functions must be defined purely as python functions.
"""


## Define loss functions

"""
This block defines a collection of loss functions. A loss function takes 4 arguments:

- **model:** *nn.neural.Module*

    The model that we are training, which for an input of size r_in produces an
    output of size r_out.


- **x:** *torch.Tensor*

    The points where the neural model is evaluated, that is, the input to the model
    used to compute the loss. It must have shape (r_in x N), where N is the number of
    points used for the loss, each one being a column in the input. For our problem,
    the input is [x; t].

    
- **y:** *torch.Tensor*

    A tensor with shape [y_size x N] used to test the output of model(x). Usually,
    we would have [nn_out x N] if we now the value that the model must produce for
    each x. However, we can also specify only voltage, or current, or use None if no
    y is used to compute the loss, like in the case of the loss of the differential
    equation.
    

- **parameters:** *dict*

    A dictionary that can be modified during setup or training, and is used to provide
    constants to the equations.
"""


def voltage_loss_function(model, x, y, parameters):
    """Squared difference of the obtained and expected voltage at the given points
    
        Simplest example of a loss function, it computes the voltage given by the model
        at points x, the expected voltage given in y, and takes the squared error. This
        is used to impose voltage at the boundaries.
    """
    return torch.square(model(x)[0,:] - y)


def full_zero_loss_function(model, x, _, parameters):
    """Squared voltage plus squared current at x

        Similar to voltage_loss_function, but it also adds the current. Can be used
        to impose clean initial conditions
    """
    y_nn = model(x)
    return torch.sum(torch.square(y_nn), 0)


def voltage_and_current_relation_loss_function(model, x, _, parameters):
    """Squared difference of the relation between the voltage and current

        This is used to impose V = I * R, usually at the receiving end of the transmission
        line. No y is specified, since the error is obtained just from voltage and current
        obtained by the model.
    """
    output = model(x)
    factor = parameters["RL"]**0.5
    return torch.square(output[0,:]/factor - output[1,:] * factor)


def voltage_and_current_loss_function(model, x, y, parameter):
    output = model(x)
    return torch.sum(torch.square(output - y), 0)


def physics_loss_function(model, x, _, parameters):
    """Imposes the voltage and current to follow telegrapher's equation

        This is where the physics enter. Taking a bunch of gradients, we obtain the error in the
        differential equations at the test points x, and square them.
    """

    y_nn = model(x)
    v = y_nn[0, :]
    i = y_nn[1, :]
    
    # Hard-coded parameters, should not stay like this!
    L = parameters["L"] # 3.0
    C = parameters["C"] # 3.0
    R = parameters["R"] # 0.01
    G = parameters["G"] # 1/10**5, eventually this should come out of the model too
    gamma = parameters["gamma"]

    grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    grad_i = torch.autograd.grad(i, x, grad_outputs=torch.ones_like(i), retain_graph=True, create_graph=True)[0]
    v_x = grad_v[0, :]
    v_t = grad_v[1, :]
    i_x = grad_i[0, :]
    i_t = grad_i[1, :]

    if gamma == 0:
        grad_v_x = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True)[0]
        grad_i_x = torch.autograd.grad(i_x, x, grad_outputs=torch.ones_like(i_x), retain_graph=True)[0]
        v_xx = grad_v_x[0,:]
        i_xx = grad_i_x[0,:]
    else:
        i_xx = 0
        v_xx = 0

    # TODO: rescaling to fix right side boundary condition
    eq_1_error = v_x + L * i_t + R * i + gamma * i_xx
    eq_2_error = i_x + C * v_t + G * v + gamma * v_xx
    
    return torch.square(eq_1_error) + torch.square(eq_2_error)



def physics_g_inference_loss_function(model, x, _, parameters):
    """Imposes the voltage and current to follow telegrapher's equation

        This is where the physics enter. Taking a bunch of gradients, we obtain the error in the
        differential equations at the test points x, and square them.
    """

    y_nn = model(x)
    v = y_nn[0, :]
    i = y_nn[1, :]
    
    # Hard-coded parameters, should not stay like this!
    L = parameters["L"] # 3.0
    C = parameters["C"] # 3.0
    R = parameters["R"] # 0.01
    g = torch.exp(torch.log(torch.tensor(10)) * model.compute_g(x[0:1, :]))
    gamma = parameters["gamma"]

    grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    grad_i = torch.autograd.grad(i, x, grad_outputs=torch.ones_like(i), retain_graph=True, create_graph=True)[0]
    v_x = grad_v[0, :]
    v_t = grad_v[1, :]
    i_x = grad_i[0, :]
    i_t = grad_i[1, :]

    if gamma == 0:
        grad_v_x = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True)[0]
        grad_i_x = torch.autograd.grad(i_x, x, grad_outputs=torch.ones_like(i_x), retain_graph=True)[0]
        v_xx = grad_v_x[0,:]
        i_xx = grad_i_x[0,:]
    else:
        i_xx = 0
        v_xx = 0

    # TODO: rescaling to fix right side boundary condition
    eq_1_error = v_x + L * i_t + R * i + gamma * i_xx
    eq_2_error = i_x + C * v_t + g * v + gamma * v_xx
    
    return torch.square(eq_1_error) + torch.square(eq_2_error)


def g_regularizer_loss_function(model, x, y, parameters):
    g = model.compute_g(x)
    return parameters["gamma"] * torch.abs(g - y)





def bc_left_setup(loss_block: neural.LossBlock, stats, epoch):
    """
        For the left_bc, we only initialize this loss by setting up the test points.
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.stack((torch.zeros(n_points), torch.linspace(0, 10, n_points)), 0)
        t = torch.linspace(0, 10, n_points)
        # loss_block.y = torch.sin(t-2) * torch.exp(-(t-2)**2)
        

        bias = loss_block.parameters["input"]["bias"]
        amplitude = loss_block.parameters["input"]["amplitude"]
        omega = loss_block.parameters["input"]["omega"]
        phase = loss_block.parameters["input"]["phase"]

        loss_block.y = amplitude * np.sin(omega * t + phase) + bias
        
        
        # loss_block.y = (1 - torch.cos(3/2 * t))/2

        loss_block.w = torch.ones(n_points)

        return


def voltage_at_right_bc_setup(loss_block: neural.LossBlock, stats, epoch):
    """
        For the right, we do something similar
    """
    print("Please do not use this function")
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.stack((torch.ones(n_points), torch.linspace(0, 10, n_points)), 0)
        loss_block.y = torch.zeros(n_points)
        loss_block.y[80:100] = 2
        loss_block.w = torch.ones(n_points)




def bc_right_setup(loss_block: neural.LossBlock, stats, epoch):
    """Just distributes points x uniformly along time in the receiving end of the transmission line
    
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.stack((torch.ones(n_points), torch.linspace(0, 10, n_points)), 0)
        loss_block.y = None
        loss_block.w = torch.ones(n_points)


def bc_from_simulation_setup(loss_block: neural.LossBlock, stats, epoch):
    """
        Impose voltage and current at both endpoints
    """
    if epoch == 0:
        import os
        import pickle
        with open(os.path.join("data", "simulator", loss_block.parameters["filename"]), "rb") as file:
            data = pickle.load(file)
            
        x_data_array = data["x_array"]
        t_data_array = data["time_array"]
        u = data["u"]
        i = data["i"]

        u_interpolator = RegularGridInterpolator([x_data_array, t_data_array], u)
        i_interpolator = RegularGridInterpolator([x_data_array, t_data_array], i)
        
        x_array = np.array(loss_block.parameters["x_values"])
        t_array = np.linspace(0, 10, loss_block.parameters["n_t_points"])

        T, X = np.meshgrid(t_array, x_array)

        reshaped_x = X.reshape(-1)
        reshaped_t = T.reshape(-1)

        all_x = np.stack((reshaped_x, reshaped_t))
        all_y = np.stack((
            np.array(u_interpolator((reshaped_x, reshaped_t))),
            np.array(i_interpolator((reshaped_x, reshaped_t)))
        ))
        
        loss_block.x = torch.tensor(all_x, dtype=torch.float)
        loss_block.y = torch.tensor(all_y, dtype=torch.float)
        loss_block.w = torch.ones(all_x.shape[1])
        


def ic_setup(loss_block: neural.LossBlock, stats, epoch):
    """Just distributes points x uniformly along the transmission line for time 0.
    
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.stack((torch.linspace(0, 1, n_points), torch.zeros(n_points)), 0)
        loss_block.y = None
        loss_block.w = torch.ones(n_points)
        



def physics_setup(loss_block: neural.LossBlock, stats: neural.TrainingStats, epoch):
    """
        They physics testing points (or residual points) are sampled from the
        domain unformly. Additionally, the can be resapled
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        time, _ = torch.sort(torch.rand(n_points) * 10)
        x = torch.rand(n_points)
        loss_block.x = torch.stack((x, time), 0)
        loss_block.x.requires_grad_(True)
        loss_block.y = None
        loss_block.w = torch.ones(loss_block.x.shape[1])
    
    else:
        resample_period = loss_block.parameters["resample_period"]
        if resample_period is not None and epoch % resample_period == 0:
            # Resamples the test points 
            n_points = loss_block.parameters["n_points"]
            time, _ = torch.sort(torch.rand(n_points) * 10)
            x = torch.rand(n_points)
            loss_block.x = torch.stack((x, time), 0)
            loss_block.x.requires_grad_(True)
            loss_block.y = None
            loss_block.w = torch.ones(loss_block.x.shape[1])
        

    if loss_block.parameters["adaptive_weighting"] is not None:
        reweighting_period = loss_block.parameters["adaptive_weighting"]["reweighting_period"]
        
        if reweighting_period % 0 == 0:
            epsilon_initial = loss_block.parameters["adaptive_weighting"]["epsilon_initial"]
            epsilon_final = loss_block.parameters["adaptive_weighting"]["epsilon_final"]
            alpha = epoch/loss_block.parameters["trainer"]["n_epochs"]
            epsilon = torch.exp(
                (1-alpha)*torch.log(torch.tensor(epsilon_initial, dtype=torch.float))
                + alpha * torch.log(torch.tensor(epsilon_final, dtype=torch.float))
            )
            print(f"Epsilon = {epsilon}")
            
            if stats.last_individual_loss["physics"] is None:
                loss_block.w = torch.ones(loss_block.x.shape[1])

            else:
                acc_loss = torch.cumsum(stats.last_individual_loss["physics"], 0)
                loss_block.w = torch.exp(-epsilon * acc_loss)



def g_regularizer_setup(loss_block: neural.LossBlock, stats: neural.TrainingStats, epoch):
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.linspace(0, 1, n_points).reshape(1, -1)
        loss_block.y = torch.ones(1, n_points) * loss_block.parameters["reference_log_g"]
        loss_block.w = torch.ones(n_points)





# Trainer setup function

"""
    The training update is similar to the loss block update, but for the training procedure.
    The state of the trainer involves stuff like the relative weight of the different losses
"""
def update_training(trainer_config: neural.TrainerConfig, stats: neural.TrainingStats, epoch):
    n_epochs = trainer_config.parameters["n_epochs"]
    alpha = epoch/n_epochs
    

    if epoch != 0 and trainer_config.parameters["adaptive_loss_weights"]:
        lr = 0.001
        total_weight = 0
        for k in trainer_config.parameters["loss_weights"]:
            loss = stats.loss_per_block[k][epoch-1]
            trainer_config.parameters["loss_weights"][k] += lr * loss
            total_weight += trainer_config.parameters["loss_weights"][k]
        
        """
        for k in trainer_config.parameters["loss_weights"]:
            trainer_config.parameters["loss_weights"][k] /= total_weight
        """
        


    if trainer_config.parameters["linear_weights"] is not None:
        for key in trainer_config.parameters["linear_weights"]["initial"].keys():
            trainer_config.parameters["loss_weights"] = {}
            trainer_config.parameters["loss_weights"][key] = \
            trainer_config.parameters["loss_weights_initial"][key] * (1-alpha) \
            + trainer_config.parameters["loss_weights_final"][key] * (alpha)



## Define setup functions
"""
Setup functions are called at every epoch. So far there are two kind of setup functions:

### Loss function setup:
They are used to set the x, y and w of their associated loss functions
(that is, the input or test points, the expected values, and the weight of each point).
Also, they set the parameters of the same loss function.

A loss function setup takes 3 arguments:

- loss_block: neural.LossBlock

    The loss block associated to the setup, which gives access to its parameters, x, y and w.


- stats: neural.TrainingStats

    Gives access to the losses obtained in the last epoch, and the overall evolution of the losses.
    This can be used to setup the weight of each training point depending on its loss in the last
    iteration.


- epoch: int

    The number of epoch.
"""