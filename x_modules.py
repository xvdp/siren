"""
Simplified versions of Siren


"""
# from collections import OrderedDict
# from modules import first_layer_sine_init, sine_init
import numpy as np
import torch
from torch import nn

# pylint: disable=no-member
class Siren(nn.Module):
    """  Simplified Siren Module
    * Disabled the Hypernetowrks for now
    * Disable the detaching and cloning coordinates - need to be passed detatched when required
    * Changed default outermost_linear = True to match example defaults
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            sine_init(final_linear, hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """ to take derivate wrt inputs, add grad before passing coordinates
        coords = coords.clone().detach().requires_grad_(True)
        """
        output = self.net(coords)
        return output

class SineLayer(nn.Module):
    """ Siren and SineLayer from explore_siren.ipynb

    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    hyperparameter.
    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if is_first:
            first_layer_sine_init(self.linear)
        else:
            sine_init(self.linear, omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

def sine_init(m, omega_0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            width = (6 / num_input)**(1/2) / omega_0
            m.weight.uniform_(-width, width)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
    
# class SimpleSiren(nn.Module):
#     """  Simplified Siren Module
#     * Disabled the Hypernetowrks for now
#     * Disable the detaching and cloning coordinates - need to be passed detatched if required
#     * Changed default outermost_linear = True to match example defaults
#     """
#     def __init__(self, in_features, hidden_features, hidden_layers, out_features,
#                  outermost_linear=True, first_omega_0=30, hidden_omega_0=30.):
#         super().__init__()

#         self.net = []
#         self.net.append(SineLayer(in_features, hidden_features,
#                                   is_first=True, omega_0=first_omega_0))

#         for i in range(hidden_layers):
#             self.net.append(SineLayer(hidden_features, hidden_features,
#                                       is_first=False, omega_0=hidden_omega_0))

#         if outermost_linear:
#             final_linear = nn.Linear(hidden_features, out_features)

#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
#                                               np.sqrt(6 / hidden_features) / hidden_omega_0)

#             self.net.append(final_linear)
#         else:
#             self.net.append(SineLayer(hidden_features, out_features, 
#                                       is_first=False, omega_0=hidden_omega_0))

#         self.net = nn.Sequential(*self.net)

#     def forward(self, coords):
#         """ to take derivate wrt inputs, add grad before passing coordinates
#         coords = coords.clone().detach().requires_grad_(True)
#         """
#         output = self.net(coords)
#         return output

#     # def forward_with_activations(self, coords, retain_grad=False):
#     #     '''Returns not only model output, but also intermediate activations.
#     #     Only used for visualizing activations later!'''
#     #     activations = OrderedDict()

#     #     activation_count = 0
#     #     x = coords.clone().detach().requires_grad_(True)
#     #     activations['input'] = x
#     #     for i, layer in enumerate(self.net):
#     #         if isinstance(layer, SineLayer):
#     #             x, intermed = layer.forward_with_intermediate(x)

#     #             if retain_grad:
#     #                 x.retain_grad()
#     #                 intermed.retain_grad()

#     #             activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
#     #             activation_count += 1
#     #         else:
#     #             x = layer(x)

#     #             if retain_grad:
#     #                 x.retain_grad()

#     #         activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
#     #         activation_count += 1

#     #     return activations


# class SineLayer(nn.Module):
#     """ Siren and SineLayer from explore_siren.ipynb

#     See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
#     If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
#     nonlinearity. Different signals may require different omega_0 in the first layer - this is a
#     hyperparameter.
#     If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
#     activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
#     """

#     def __init__(self, in_features, out_features, bias=True,
#                  is_first=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = omega_0
#         self.is_first = is_first

#         self.in_features = in_features
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         self.init_weights()

#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features,
#                                              1 / self.in_features)
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
#                                              np.sqrt(6 / self.in_features) / self.omega_0)

#     def forward(self, input):
#         return torch.sin(self.omega_0 * self.linear(input))

#     # def forward_with_intermediate(self, input):
#     #     # For visualization of activation distributions
#     #     intermediate = self.omega_0 * self.linear(input)
    #     return torch.sin(intermediate), intermediate


# ###
# # from modules
# #
# class SineSiren(nn.Module):
#     ''' Simplified BVP
#     '''

#     def __init__(self, out_features=1, in_features=2, hidden_features=256, num_hidden_layers=3, **kwargs):
#         super().__init__()
#         #type='sine', 
#         self.mode = 'mlp'

#         self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
#                            hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
#         # print(self)

#     def forward(self, model_input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # Enables us to compute gradients w.r.t. coordinates
#         # this duplicates memory unnecessarily if passed as cuda
#         # clone detach on dataloader not here.
#         # coords = model_input['coords'].clone().detach().requires_grad_(True)
#         coords = model_input['coords']

#         output = self.net(coords, get_subdict(params, 'net'))
#         return {'model_in': coords, 'model_out': output}

#     def forward_with_activations(self, model_input):
#         '''Returns not only model output, but also intermediate activations.'''
#         coords = model_input['coords'].clone().detach().requires_grad_(True)
#         activations = self.net.forward_with_activations(coords)
#         return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


    
# class Sine(nn.Module):
#     def __init(self):
#         super().__init__()

#     def forward(self, input):
#         # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
#         return torch.sin(30 * input)


# class FCBlock(nn.Module):
#     '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
#     Can be used just as a normal neural network though, as well.
#     '''

#     def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
#                  outermost_linear=False, nonlinearity='relu', weight_init=None):
#         super().__init__()

#         self.first_layer_init = None


#         # sine
#         nl = Sine()
#         nl_weight_init = sine_init
#         first_layer_init = first_layer_sine_init
#         # sigmoid
#         # nl = nn.Sigmoid()
#         # nl_weight_init = init_weights_xavier
#         # first_layer_init = None

#         if weight_init is not None:  # Overwrite weight init if passed
#             self.weight_init = weight_init
#         else:
#             self.weight_init = nl_weight_init

#         self.net = []
#         self.net.append(MetaSequential(
#             BatchLinear(in_features, hidden_features), nl
#         ))

#         for i in range(num_hidden_layers):
#             self.net.append(MetaSequential(
#                 BatchLinear(hidden_features, hidden_features), nl
#             ))

#         if outermost_linear:
#             self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
#         else:
#             self.net.append(MetaSequential(
#                 BatchLinear(hidden_features, out_features), nl
#             ))

#         self.net = MetaSequential(*self.net)
#         if self.weight_init is not None:
#             self.net.apply(self.weight_init)

#         if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
#             self.net[0].apply(first_layer_init)

#     def forward(self, coords, params=None, **kwargs):
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         output = self.net(coords, params=get_subdict(params, 'net'))
#         return output





# ########################
# # Initialization methods
# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     # For PINNet, Raissi et al. 2019
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     # grab from upstream pytorch branch and paste here for now
#     def norm_cdf(x):
#         # Computes standard normal cumulative distribution function
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.

#     with torch.no_grad():
#         # Values are generated by using a truncated uniform distribution and
#         # then using the inverse CDF for the normal distribution.
#         # Get upper and lower cdf values
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)

#         # Uniformly fill tensor with values from [l, u], then translate to
#         # [2l-1, 2u-1].
#         tensor.uniform_(2 * l - 1, 2 * u - 1)

#         # Use inverse cdf transform for normal distribution to get truncated
#         # standard normal
#         tensor.erfinv_()

#         # Transform to proper mean, std
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)

#         # Clamp to ensure it's in the proper range
#         tensor.clamp_(min=a, max=b)
#         return tensor


# def init_weights_trunc_normal(m):
#     # For PINNet, Raissi et al. 2019
#     # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             fan_in = m.weight.size(1)
#             fan_out = m.weight.size(0)
#             std = math.sqrt(2.0 / float(fan_in + fan_out))
#             mean = 0.
#             # initialize with the same behavior as tf.truncated_normal
#             # "The generated values follow a normal distribution with specified mean and
#             # standard deviation, except that values whose magnitude is more than 2
#             # standard deviations from the mean are dropped and re-picked."
#             _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


# def init_weights_normal(m):
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


# def init_weights_selu(m):
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


# def init_weights_elu(m):
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


# def init_weights_xavier(m):
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             nn.init.xavier_normal_(m.weight)


# def sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             # See supplement Sec. 1.5 for discussion of factor 30
#             m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)



# def first_layer_sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
#             m.weight.uniform_(-1 / num_input, 1 / num_input)
