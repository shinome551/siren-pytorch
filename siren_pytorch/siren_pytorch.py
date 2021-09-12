import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'n d -> n () d')

        return self.last_layer(x)

# modulatory feed forward

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), 1)

        return tuple(hiddens)

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.setGrid(image_width, image_height)
        
        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )      

    def setGrid(self, image_width, image_height):
        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> () (h w) c')
        self.register_buffer('grid', mgrid)
        self.image_width = image_width
        self.image_height = image_height
    
    def forward(self, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        ## tuple pf (n, mod_d)
        mods = self.modulator(latent) if modulate else None

        batch_size = latent.size(0) if exists(latent) else 1
        coords = self.grid.expand(batch_size, -1, -1).clone()
        out = self.net(coords, mods)
        out = rearrange(out, 'n (h w) c -> n c h w', n = batch_size ,h = self.image_height, w = self.image_width)

        return out
