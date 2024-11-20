import torch
import neural
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


def get_derivative_function(n_x, R1, G, L, C, RL, input_function):
    def get_derivative(t, x):
        v = x[:n_x]
        i = x[n_x:]
        
        """
        if t <= 2  and t >= 1:
            u = 1
        else:
            u = 0
        """
        # u = np.sin(5/8 * t)
        # u = (1 - np.cos(3/2*t))+1
        u = input_function(t)

        v[0] = u
        i[-1] = v[-1]/RL
        
        der_v = np.zeros(v.shape)
        der_i = np.zeros(i.shape)
        
        der_v[1:] = 1/C* (i[:-1] - i[1:] - v[1:] * G[1:])
        der_i[:-1] = 1/L* (-v[1:] + v[:-1] - i[:-1] * R1)
        
        return np.concatenate((der_v, der_i))

    return get_derivative


class TransmissionLineParameters:
    def __init__(self, R1o, R2o, Lo, Co, RL):
        self.R1o = R1o
        self.R2o = R2o
        self.Lo = Lo
        self.Co = Co
        self.RL = RL

    @classmethod
    def default(cls):
        R1o = 0.01
        R2o = 1e1
        Lo = 0.3
        Co = 0.3
        RL = 5.0

        return cls(R1o, R2o, Lo, Co, RL)


    def simulate(self, n_x, g_factor_function, input_function):
        x_array = np.linspace(0, 1, n_x)
        dx = 1/n_x
        R1 = self.R1o * dx
        R2 = self.R2o / dx
        L = self.Lo * dx
        C = self.Co * dx
        RL = self.RL

        g_factor_array = g_factor_function(x_array)

        G = 1/R2 * g_factor_array
        

        sol = solve_ivp(
            get_derivative_function(n_x, R1, G, L, C, RL, input_function),
            [0, 1],
            np.zeros(2*n_x),
            method='RK45',
            dense_output=True,
            rtol=1e-9,
            atol=1e-9
        )

        return TransmissionLineSimulation(
            self,
            n_x,
            g_factor_array,
            input_function(sol.t),
            sol
        )


class TransmissionLineSimulation:
    def __init__(
        self,
        parameters: TransmissionLineParameters,
        n_x,
        g_factor_array,
        input_array,
        sol
    ):
        self.parameters = parameters
        self.n_x = n_x
        self.t_array = sol.t
        self.x_array = np.linspace(0, 1, n_x)
        self.input = input_array
        self.g_factor_array = g_factor_array
        self.v_grid = sol.y[:n_x, :]
        self.i_grid = sol.y[n_x:, :]


    def save(self, filename):
        torch.save(self, filename)


    @classmethod
    def load(cls, filename):
        return torch.load(filename)


    def sample(self, x_array, t_array, mode):
        if mode == "voltage":
            value = self.v_grid
        elif mode == "current":
            value = self.i_grid

        interpolator = RegularGridInterpolator((self.x_array, self.t_array), value)

        X, T = np.meshgrid(x_array, t_array)
        
        x_array = X.reshape(-1)
        t_array = T.reshape(-1)

        input = np.stack((x_array, t_array))

        interpolated_value = interpolator((x_array, t_array))

        return (input, interpolated_value)
    



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

# Data-based loss functions

def full_zero_loss_function(model, x, _, parameters):
    """Squared voltage plus squared current at x

        Similar to voltage_loss_function, but it also adds the current. Can be used
        to impose clean initial conditions
    """
    y_nn = model(x)
    return torch.sum(torch.square(y_nn), 0)


def voltage_measurement_loss_function(model, x, y, parameter):
    model_voltage = model(x)[0, :]
    return torch.square(model_voltage - y)


def current_measurement_loss_function(model, x, y, parameter):
    model_current = model(x)[1, :]
    return torch.square(model_current - y)



# Physics-based loss functions

def voltage_current_relation_loss_function(model, x, _, parameters):
    """Squared difference of the relation between the voltage and current

        This is used to impose V = I * R, usually at the receiving end of the transmission
        line. No y is specified, since the error is obtained just from voltage and current
        obtained by the model.
    """
    output = model(x)
    factor = parameters["RL"]**0.5
    return torch.square(output[0,:]/factor - output[1,:] * factor)


def physics_loss_function(model, x, _, parameters):
    """Imposes the voltage and current to follow telegrapher's equation

        This is where the physics enter. Taking a bunch of gradients, we obtain the error in the
        differential equations at the test points x, and square them.
    """

    y_nn = model(x)
    v = y_nn[0, :]
    i = y_nn[1, :]
    
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



# Setup functions

def ic_setup(loss_block: neural.LossBlock, stats, epoch):
    """Just distributes points x uniformly along the transmission line for time 0.
    
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_x_points"]
        loss_block.x = torch.stack((torch.linspace(0, 1, n_points), torch.zeros(n_points)), 0)
        loss_block.y = None
        loss_block.w = torch.ones(n_points)


def voltage_measurement_setup(loss_block: neural.LossBlock, stats, epoch):
    """
        Impose voltage and current at both endpoints
    """
    if epoch == 0:
        x_array = np.array(loss_block.parameters["x_points"])
        t_array = np.linspace(0, 1, loss_block.parameters["n_t_points"])

        (input_array, output_array) = TransmissionLineSimulation.load(
            loss_block.parameters["filename"]
        ).sample(x_array, t_array, mode="voltage")

        loss_block.x = torch.tensor(input_array, dtype=torch.float)
        loss_block.y = torch.tensor(output_array, dtype=torch.float)
        loss_block.w = torch.ones(input_array.shape[1])


def current_measurement_setup(loss_block: neural.LossBlock, stats, epoch):
    """
        Impose voltage and current at both endpoints
    """
    if epoch == 0:
        x_array = np.array(loss_block.parameters["x_points"])
        t_array = np.linspace(0, 1, loss_block.parameters["n_t_points"])

        (input_array, output_array) = TransmissionLineSimulation.load(
            loss_block.parameters["filename"]
        ).sample(x_array, t_array, mode="current")

        loss_block.x = torch.tensor(input_array, dtype=torch.float)
        loss_block.y = torch.tensor(output_array, dtype=torch.float)
        loss_block.w = torch.ones(input_array.shape[1])



def voltage_current_relation_setup(loss_block: neural.LossBlock, stats, epoch):
    """Just distributes points x uniformly along time in the receiving end of the transmission line
    
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        loss_block.x = torch.stack((torch.ones(n_points), torch.linspace(0, 10, n_points)), 0)
        loss_block.y = None
        loss_block.w = torch.ones(n_points)


def physics_setup(loss_block: neural.LossBlock, stats: neural.TrainingStats, epoch):
    """
        They physics testing points (or residual points) are sampled from the
        domain unformly. Additionally, the can be resapled
    """
    if epoch == 0:
        n_points = loss_block.parameters["n_points"]
        time, _ = torch.sort(torch.rand(n_points))
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
            time, _ = torch.sort(torch.rand(n_points))
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