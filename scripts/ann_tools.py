import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as functional
import numpy as np
import os
import xarray as xr
from time import time
from escnn import gspaces
from escnn import nn as nn_escnn
from sklearn.model_selection import train_test_split
import json

def tensor_from_xarray(x, torch_type=torch.float32):
    if isinstance(x, xr.DataArray):
        return torch.tensor(x.to_numpy()).type(torch_type)
    elif isinstance(x, torch.Tensor):
        return x.type(torch_type)

def torch_pad(x, one_side_pad = 1,
                      left=False, right=False, top=False, bottom=False):
    '''
    x is the torch tensor of size Ny x Nx
    Here we implement padding with circular B.C.
    for zonal (Nx) direction and zero B.C. for meridional (Ny)

    By default, we do not do padding
    '''
    ny, nx = x.shape

    # Nothing to pad
    if one_side_pad == 0:
        return x
    
    # Compute size of the resulting array
    Nx = nx + (int(left) + int(right)) * one_side_pad
    Ny = ny + (int(top) + int(bottom)) * one_side_pad
    y = torch.zeros((Ny,Nx), dtype=x.dtype)

    # Copy original array to the center
    x_start = one_side_pad if left else 0
    y_start = one_side_pad if bottom else 0

    y[y_start:y_start+ny,x_start:x_start+nx] = x

    if top:
        y[-one_side_pad:,:] = 0.
    if bottom:
        y[:one_side_pad,:] = 0.
    
    if left:
        y[y_start:y_start+ny,:one_side_pad] = x[:,-one_side_pad:]
    if right:
        y[y_start:y_start+ny,-one_side_pad:] = x[:,:one_side_pad]

    return y

def image_to_3x3_stencil(x):
    '''
    Note that x is the fast direction in CM2.6 and 
    also it is the rightmost.
    Transforms image data to Npoints * 9 data. 
    The input and output are torch tensors
    '''
    ny, nx = x.shape
    # Single feature has dimension Npoints x 9,
    # where 9 corresponds to stencil 3x3, and 
    # zonal direction is the fast one
    y = torch.zeros((nx-2)*(ny-2),9)
    k = 0
    for j in range(1,x.shape[0]-1):
        for i in range(1,x.shape[1]-1):
            y[k,:] = x[j-1:j+2,i-1:i+2].reshape(1,9)
            k += 1
    return y

def image_to_nxn_stencil_gpt(x, stencil_size=3,
                             rotation=0, reflect_x=False, reflect_y=False):
    '''
    Extension of function above with arbitrary stencil size
    stencil_size x stencil_size
    
    The rotation parameter allows to rotate input stencil
    by 0, 90, 180, 270 degrees conunter-clockwise
    '''
    n = stencil_size
    if rotation == 0:
        y = x.unfold(0,n,1).unfold(1,n,1)
    elif rotation == 90:
        y = x.unfold(1,n,1).flip(-1).unfold(0,n,1)
    elif rotation == 180:
        y = x.unfold(0,n,1).flip(-1).unfold(1,n,1).flip(-1)
    elif rotation == 270:
        y = x.unfold(1,n,1).unfold(0,n,1).flip(-1)
    else:
        print('Error: use rotation one of 0, 90, 180, 270')
        
    if reflect_x:
        y = y.flip(-1)
    if reflect_y:
        y = y.flip(-2) 
        
    return y.reshape(-1,n*n)

def log_to_xarray(log_dict):
    anykey = list(log_dict.keys())[0]
    num_epochs = len(log_dict[anykey])
    epoch = coord(np.arange(1, num_epochs+1), 'epoch')
    for key in log_dict.keys():
        log_dict[key] = xr.DataArray(log_dict[key], dims='epoch', coords=[epoch])
        
    return xr.Dataset(log_dict)

class ANN(nn.Module):
    def __init__(self, layer_sizes=[3, 17, 27, 5]):
        super().__init__()
        
        self.layer_sizes = layer_sizes

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers)-1:
                x = functional.relu(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def compute_loss(self, x, ytrue):
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}

class ANN_equivariant(nn.Module):
    def __init__(self, hidden_layer_size=16, stencil_size=3, symmetry='flipRot2dOnR2_N4'):
        '''
        As this ANN is equivariant, it strongly assumes order 
        and meaning of all inputs and outputs.
        We are having 
        ['sh_xy', 'sh_xx', 'vort_xy'] inputs
        on a stencil fo size given by stencil_size
        The outputs are 
        [Txy, 0.5*(Txx-Tyy), 0.5*(Txx+Tyy)]
        which are later transformed to 
        [Txy, Txx, Tyy]
        for compatibility with standard ANN
        '''
        super().__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.stencil_size = stencil_size
        self.symmetry = symmetry

        if symmetry == 'flipRot2dOnR2_N4':
            self.number_of_equivariant_hidden_neurons = hidden_layer_size//8
            r2_act = gspaces.flipRot2dOnR2(N=4)
            # [sh_xy, sh_xx, vort_xy]
            self.feat_type_in  = nn_escnn.FieldType(r2_act, [r2_act.irrep(1,2)] + [r2_act.irrep(0,2)] + [r2_act.irrep(1,0)])
            # [Txy, 0.5*(Txx-Tyy), 0.5*(Txx+Tyy)]
            self.feat_type_out = nn_escnn.FieldType(r2_act, [r2_act.irrep(1,2)] +       
                                                [r2_act.irrep(0,2)] + 
                                                [r2_act.irrep(0,0)])
        elif symmetry == 'flipRot2dOnR2_N8':
            self.number_of_equivariant_hidden_neurons = hidden_layer_size//16
            r2_act = gspaces.flipRot2dOnR2(N=8)
            # [sh_xx, sh_xy, vort_xy]
            self.feat_type_in  = nn_escnn.FieldType(r2_act, [r2_act.irrep(1,2)] + [r2_act.irrep(1,0)])
            # [0.5*(Txx-Tyy), Txy, 0.5*(Txx+Tyy)]
            self.feat_type_out = nn_escnn.FieldType(r2_act, [r2_act.irrep(1,2)] + [r2_act.irrep(0,0)])
        else:
            raise NotImplementedError(f"We support only two symmetries: flipRot2dOnR2_N4 and flipRot2dOnR2_N8 but not {symmetry}") 

        # Type of hidden neurons is the same
        self.feat_type_hid = nn_escnn.FieldType(r2_act, self.number_of_equivariant_hidden_neurons*[r2_act.regular_repr])

        self.model = nn_escnn.SequentialModule(
            nn_escnn.R2Conv(self.feat_type_in, self.feat_type_hid, kernel_size=stencil_size),
            nn_escnn.ReLU(self.feat_type_hid),
            nn_escnn.R2Conv(self.feat_type_hid, self.feat_type_out, kernel_size=1)
        ).to().eval()
    
    def forward(self, _x):
        '''
        Here we assume standard input vector of size 27 as we use in ANN
        where 27 is (3 channels)*(3 y_points)*(3 x_points)
        '''
        # Reshape ANN input vector to a format compatible with CNN
        x = _x.view(-1, 3, self.stencil_size, self.stencil_size)

        if self.symmetry == 'flipRot2dOnR2_N8':
            # Swap first two channels compared to a regular ordering
            x0 = x[:,1]
            x1 = x[:,0]
            x2 = x[:,2]
            x = torch.stack([x0,x1,x2], dim=1)

        x = self.feat_type_in(x)
        
        _y = self.model(x).tensor.squeeze(-1).squeeze(-1)

        if self.symmetry == 'flipRot2dOnR2_N4':
            # Transform from Txy, 0.5*(Txx-Tyy), 0.5*(Txx+Tyy)]
            # back to [Txy, Txx, Tyy]
            y0 = _y[:, 0]
            y1 = + _y[:, 1] + _y[:, 2]
            y2 = - _y[:, 1] + _y[:, 2]
            y = torch.stack([y0, y1, y2], dim=1)
        elif self.symmetry == 'flipRot2dOnR2_N8':
            # Transform from 0.5*(Txx-Tyy), Txy, 0.5*(Txx+Tyy)
            # back to [Txy, Txx, Tyy]
            y0 = _y[:, 1]
            y1 = + _y[:, 0] + _y[:, 2]
            y2 = - _y[:, 0] + _y[:, 2]
            y = torch.stack([y0, y1, y2], dim=1)

        return y
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def compute_loss(self, x, ytrue):
        return {'loss': nn.MSELoss()(self.forward(x), ytrue)}

def equivariant_to_regular_ANN(ann_equivariant):
    ann = ANN(layer_sizes = [3 * ann_equivariant.stencil_size**2, ann_equivariant.hidden_layer_size, 3])
    
    input_layer = ann_equivariant.model[0]
    output_layer = ann_equivariant.model[2]

    if ann_equivariant.symmetry == 'flipRot2dOnR2_N4':
        input_transformation = torch.eye(27,27)
    elif ann_equivariant.symmetry == 'flipRot2dOnR2_N8':
        input_transformation = torch.zeros(27,27)
        # Swap first two variables
        n = ann_equivariant.stencil_size**2
        for i in range(n):
            input_transformation[i,i+n] = 1
            input_transformation[i+n,i] = 1
            # Keep the third variable without change
            input_transformation[i+2*n,i+2*n] = 1

    # Reshape is needed to transform CNN to ANN
    ann.layers[0].weight.data = input_layer.expand_parameters()[0].reshape(ann_equivariant.hidden_layer_size, -1) @ input_transformation
    ann.layers[0].bias.data   = input_layer.expand_parameters()[1]

    if ann_equivariant.symmetry == 'flipRot2dOnR2_N4':
        # Transform from Txy, 0.5*(Txx-Tyy), 0.5*(Txx+Tyy)]
        # back to [Txy, Txx, Tyy]
        output_transformation = torch.eye(3,3)
        output_transformation[1,2] = 1
        output_transformation[2,1] = -1
    elif ann_equivariant.symmetry == 'flipRot2dOnR2_N8':
        # Transform from 0.5*(Txx-Tyy), Txy, 0.5*(Txx+Tyy)
        # back to [Txy, Txx, Tyy]
        output_transformation = torch.zeros(3,3)
        output_transformation[0,1] = 1
        output_transformation[1,0] = 1
        output_transformation[1,2] = 1
        output_transformation[2,0] = -1
        output_transformation[2,2] = +1
    
    # squeeze is required to transform 1-point CNN to ANN
    ann.layers[1].weight.data = output_transformation @ output_layer.expand_parameters()[0].squeeze()
    ann.layers[1].bias.data   = output_transformation @ output_layer.expand_parameters()[1]

    return ann

def regular_to_equivariant_ANN(ann, stencil_size = 3,
    hidden_layer_size = None,
    nsamples=1000000, num_epochs=10, batch_size=1000, learning_rate=0.01):

    if len(ann.layers) != 2:
        raise NotImplementedError("Current version must have only two layers. It is possible to extend it")

    if hidden_layer_size is None:
        hidden_layer_size=8*(ann.layer_sizes[1]//8)
    ann_equivariant = ANN_equivariant(hidden_layer_size=hidden_layer_size, stencil_size=stencil_size)

    # We find the optimal equivariant ANN which is as close as possible to the regular ANN
    # by performing gradient descent on input vectors of norm 1 (as it happens in our Perezhogin 2025 et al paper)
    X = np.random.randn(nsamples, 3 * stencil_size**2).astype('float32')
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    Y = ann(torch.tensor(X)).detach().numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    train(ann_equivariant, X_train, Y_train, X_test, Y_test,
          num_epochs = num_epochs, batch_size = batch_size, learning_rate = learning_rate)

    return ann_equivariant 
    
def export_ANN(ann, input_norms, output_norms, filename='ANN_test.nc'):
    if isinstance(ann, ANN_equivariant):
        '''
        In this case we additionally save ANN in equivariant form
        '''
        print('Saving weights of equaivariant part of ANN as .nc')
        parameters = xr.Dataset()
        parameters['weights1'] = xr.DataArray(ann.model[0].weights.detach().numpy(), dims='pdim1')
        parameters['weights2'] = xr.DataArray(ann.model[2].weights.detach().numpy(), dims='pdim2')
        parameters['biases1'] = xr.DataArray(ann.model[0].bias.detach().numpy(), dims='pdim3')
        parameters['biases2'] = xr.DataArray(ann.model[2].bias.detach().numpy(), dims='pdim4')
        netcdf_eANN = os.path.join(os.path.dirname(filename), "eANN.nc")
        print('netcdf_eANN', netcdf_eANN)
        parameters.to_netcdf(netcdf_eANN)

        print('Saving additionally weights of equivariant part of ANN as .pth')
        torch_file = os.path.join(os.path.dirname(filename), "Tall.pth")
        json_file = os.path.join(os.path.dirname(filename), "Tall.json")
        torch.save(ann.state_dict(), torch_file)
        params = dict(hidden_layer_size=ann.hidden_layer_size, stencil_size=ann.stencil_size, symmetry=ann.symmetry)
        with open(json_file, "w") as f:
            json.dump(params, f, indent=4)
        
        # Save also ANN in regular format
        ann = equivariant_to_regular_ANN(ann)

    ds = xr.Dataset()
    ds['num_layers'] = xr.DataArray(len(ann.layer_sizes)).expand_dims('dummy_dimension')
    ds['layer_sizes'] = xr.DataArray(ann.layer_sizes, dims=['nlayers'])
    ds = ds.astype('int32') # MOM6 reads only int32 numbers
    
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = f'A{i}'
        bias = f'b{i}'
        ncol = f'ncol{i}'
        nrow = f'nrow{i}'
        layer = ann.layers[i]
        
        # Transposed, because torch is row-major, while Fortran is column-major
        ds[matrix] = xr.DataArray(layer.weight.data.T, dims=[ncol, nrow])
        ds[bias] = xr.DataArray(layer.bias.data, dims=[nrow])
    
    # Save true answer for random vector for testing
    x0 = torch.randn(ann.layer_sizes[0])
    y0 = ann(x0 / input_norms) * output_norms
    nrow = f'nrow{len(ann.layers)-1}'
    
    ds['x_test'] = xr.DataArray(x0.data, dims=['ncol0'])
    ds['y_test'] = xr.DataArray(y0.data, dims=[nrow])
    
    ds['input_norms']  = xr.DataArray(input_norms.data, dims=['ncol0'])
    ds['output_norms'] = xr.DataArray(output_norms.data, dims=[nrow])

    # print('x_test = ', ds['x_test'].data)
    # print('y_test = ', ds['y_test'].data)
    
    if os.path.exists(filename):
        print(f'Rewrite {filename} ?')
        input()
        os.system(f'rm -f {filename}')
        print(f'{filename} is rewritten')
    
    ds.to_netcdf(filename)
    
def import_ANN(filename='ANN_test.nc', return_both=False):
    ds = xr.open_dataset(filename)
    layer_sizes = ds['layer_sizes'].values

    ann = ANN(layer_sizes)
    
    for i in range(len(ann.layers)):
        # Naming convention for weights and dimensions
        matrix = torch.tensor(ds[f'A{i}'].T.values) # Transpose back: it is convention
        bias = torch.tensor(ds[f'b{i}'].values)
        ann.layers[i].weight.data = matrix
        ann.layers[i].bias.data = bias
    try:    
        x_test = torch.tensor(ds['x_test'].values.reshape(1,-1))
        y_pred = ann(x_test).detach()
        y_test = ds['y_test'].values
        
        rel_error = float(np.abs(y_pred - y_test).max() / np.abs(y_test).max())
        if rel_error > 1e-6:
            print(f'Test prediction using {filename}: {rel_error}')
    except:
        pass

    # Import equivariant ANN if torch file is available
    torch_file = os.path.join(os.path.dirname(filename), "Tall.pth")
    json_file = os.path.join(os.path.dirname(filename), "Tall.json")
    if os.path.exists(torch_file):
        print('Returning equivariant ANN instead')
        with open(json_file, "r") as f:
            params = json.load(f)
        print(params)
        ann_equivariant = ANN_equivariant(**params)
        ann_equivariant.load_state_dict(torch.load(torch_file))
        if return_both:
            return ann, ann_equivariant
        else:
            return ann_equivariant
    return ann

class AverageLoss():
    '''
    Accumulates dictionary of losses over batches
    and computes mean for epoch.
    List of keys to accumulate given by 'losses'
    Usage:
    Init before epoch. 
    Accumulate over batches for given epoch
    Average after epoch
    '''
    def __init__(self, log_dict):
        self.init_me = True
        self.count = {}

    def accumulate(self, log_dict, losses, n: int):
        '''
        log_dict: dictionary of timeseries
        losses: dictionary of loss on a batch
        n: number of elements in batch
        '''
        keys = losses.keys()
        if (self.init_me):
            new_keys = set(losses.keys())-set(log_dict.keys())
            for key in new_keys:
                log_dict[key] = []
            for key in keys:
                self.count[key] = 0
                log_dict[key].append(0.)
            self.init_me = False

        for key in keys:
            value = losses[key]
            # extract floats from scalar tensors
            if isinstance(value, torch.Tensor):
                try:
                    value = value.item()
                except:
                    value = value.cpu().numpy()
            log_dict[key][-1] += value * n
            self.count[key] += n
    
    def average(self, log_dict):
        '''
        Updates last element of dictionary with 
        average value
        '''
        for key in self.count.keys():
            log_dict[key][-1] = log_dict[key][-1] / self.count[key]

def dict_postfix(mydict, postfix):
    return {str(key)+postfix: val for key, val in mydict.items()}

def minibatch(*arrays, batch_size=64, shuffle=True):
    '''
    Recieves arbitrary number of numpy arrays
    of size 
    Nbatch x Nfeatures x Ny x Nx.
    Returns multiple batches of tensors of size 
    batch_size x Nfeatures x Ny x Nx.
    '''
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
        np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        yield tuple(torch.as_tensor(array[idx]) if isinstance(array, np.ndarray) else array[idx] for array in arrays)
        

def evaluate_test(net, *arrays: np.array, batch_size=64, postfix='_test', device=None):
    '''
    Updates logger on test dataset
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # To do not update parameters of 
    # Batchnorm2d layers
    net.eval()

    logger = AverageLoss(net.log_dict)
    for xy in minibatch(*arrays, batch_size=batch_size):
        with torch.no_grad():
            losses = net.compute_loss(*[x.to(device) for x in xy])
        logger.accumulate(net.log_dict, dict_postfix(losses, postfix), len(xy[0]))
    
    logger.average(net.log_dict)
    net.train()

def train(net, X_train: np.array, Y_train: np.array, 
        X_test: np.array, Y_test: np.array, 
        num_epochs, batch_size, learning_rate, device=None,
        normalize_loss=False, print_frequency=1):
    '''
    X_train, Y_train are arrays of size
    Nbatch x Nfeatures x Ny x Nx.
    For this function to use, class 'net'
    should implement function compute_loss(x,y) returning 
    dictionary, where key 'loss'
    is used for optimization,
    while others are used for logger.
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    else:
        device_name = "cpu"
    
    if print_frequency > 0:
        print(f"Training starts on device {device_name}, number of samples {len(X_train)}")
    
    net.to(device)
    # Switch batchnorm2d layer to training mode
    net.train()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)  

    try:
        net.log_dict
    except:
        net.log_dict = {}
            
    t_s = time()
    for epoch in range(0,num_epochs):
        t_e = time()
        logger = AverageLoss(net.log_dict)
        for x, y in minibatch(X_train, Y_train, batch_size=batch_size):
        # for x, y in data_loader:
            optimizer.zero_grad()
            losses = net.compute_loss(x.to(device),y.to(device))
            if normalize_loss:
                losses['loss'] = losses['loss'] / losses['loss'].detach()
            losses['loss'].backward() # optimize over the 'loss' value
            optimizer.step()
            logger.accumulate(net.log_dict, losses, len(x))
        scheduler.step()

        logger.average(net.log_dict)
        evaluate_test(net, X_test, Y_test, batch_size=batch_size, device=device)
        t = time()
        if epoch % print_frequency == 0 and print_frequency > 0:
            print('[%d/%d] [%.2f/%.2f] Loss: [%.6f, %.6f]' 
                % (epoch+1, num_epochs,
                t-t_e, (t-t_s)*(num_epochs/(epoch+1)-1),
                net.log_dict['loss'][-1], net.log_dict['loss_test'][-1]))
