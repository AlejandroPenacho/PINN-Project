import numpy as np
import torch as torch
import matplotlib.pyplot as plt

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
        