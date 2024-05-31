
#####################################   Imports   #################################################
import datetime
import argparse
import os
import time

import torch
torch.set_default_dtype(torch.float32)
import torch.nn.functional as F
import numpy as np
import sys
from nflows import transforms, distributions, flows

from data import get_dataloader
from data import save_samples_to_file, save_calib_to_file, save_dedx_to_file

#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# usage modes
parser.add_argument('--training', action='store_true', help='train calib')
parser.add_argument('--generate_to_file', action='store_true', help='generate from a trained flow and save to file')
parser.add_argument('--save_pt', action='store_true', help='')
parser.add_argument('--saveONNX', action='store_true', help='')
parser.add_argument('--onnx_file_path', default='', help='')
parser.add_argument('--check_pt', action='store_true', help='')
parser.add_argument('--restore', action='store_true', help='restore and train a flow')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int, help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--gen_dir', default='', help='Where to store the generated file')
parser.add_argument('--output_file', default='default.hdf5', help='')
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--restore_file', type=str, default=None, help='Model file to restore.')
parser.add_argument('--data_dir', default='', help='Where to find the training dataset')
parser.add_argument('--use_test_dataloader', action='store_true', help='')

parser.add_argument('--particle_type', '-p', help='Which particle, "e+", "eplus", or "piplus"')
parser.add_argument('--num_feature', default=1, type=int, help='How many features are trained')
parser.add_argument('--num_block' , default=2, type=int, help='')
parser.add_argument('--hidden_features' , default=64, type=int, help='')
parser.add_argument('--num_epochs', default=100, type=int, help='How many epochs are trained')
parser.add_argument('--gen_events', default=100, type=int, help='How many events are generated')

# MAF parameters
parser.add_argument('--n_blocks', type=str, default='8',
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--hidden_size_multiplier', type=int, default=None,
                    help='Hidden layer size for each MADE block in an MAF'+\
                    ' is given by the dimension times this factor.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--batch_norm', action='store_true', default=False,
                    help='Use batch normalization')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--use_residual', action='store_true', default=False,
                    help='Use residual layers in the NNs')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.05,
                    help='dropout probability')
parser.add_argument('--tail_bound', type=float, default=14., help='Domain of the RQS')
parser.add_argument('--cond_base', action='store_true', default=False,
                    help='Use Gaussians conditioned on energy as base distribution.')
parser.add_argument('--init_id', action='store_true',
                    help='Initialize Flow to be identity transform')

# training params
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--num_try', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=100,
                    help='How often to show loss statistics and save samples.')

parser.add_argument('--workpath', type=str, default='.', help='work path ')
parser.add_argument('--save_model_name', type=str, default='.', help='save_model_name ')
parser.add_argument('--schedulerType', type=int, default=0)
parser.add_argument('--input_lr', type=float, default=0.0, help='')
parser.add_argument('--w_decay', type=float, default=0, help='')
parser.add_argument('--trunc', type=float, default=0, help='')
parser.add_argument('--gen_batch', type=int, default=1000)
parser.add_argument('--pt_file_path', type=str, default='.', help='args.pt_file_path ')
parser.add_argument('--check_pt_file', type=str, default='.', help=' ')

#######################################   helper functions   #######################################
class LRWarmUPSF(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, threshold, sf):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.previous_loss = 9999
        self.change_threshold = threshold
        self.sf = sf
    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr
    def sf_learning_rate(self, loss):
        if (self.previous_loss - loss) > self.change_threshold:
            pass
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr']*self.sf
        self.previous_loss = loss

    def step(self, cur_iteration, loss):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.sf_learning_rate(loss)

# used in transformation between energy and logit space:
# (should match the ALPHA in data.py)
ALPHA = 1e-6

def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((torch.sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp_(clamp_low, clamp_high)

def one_hot(values, num_bins):
    """ one-hot encoding of values into num_bins """
    # values are energies in [0, 1], need to be converted to integers in [0, num_bins-1]
    values *= num_bins
    values = values.type(torch.long)
    ret = F.one_hot(values, num_bins)
    return ret.squeeze().double()

def one_blob(values, num_bins):
    """ one-blob encoding of values into num_bins, cf sec. 4.3 of 1808.03856 """
    # torch.tile() not yet in stable release, use numpy instead
    values = values.cpu().numpy()[..., np.newaxis]
    y = np.tile(((0.5/num_bins) + np.arange(0., 1., step=1./num_bins)), values.shape)
    res = np.exp(((-num_bins*num_bins)/2.)
                 * (y-values)**2)
    res = np.reshape(res, (-1, values.shape[-1]*num_bins))
    return torch.tensor(res)

def remove_nans(tensor):
    """removes elements in the given batch that contain nans
       returns the new tensor and the number of removed elements"""
    tensor_flat = tensor.flatten(start_dim=1)
    good_entries = torch.all(tensor_flat == tensor_flat, axis=1)
    res_flat = tensor_flat[good_entries]
    tensor_shape = list(tensor.size())
    tensor_shape[0] = -1
    res = res_flat.reshape(tensor_shape)
    return res, len(tensor) - len(res)


def generate_to_file(args, num_events, sim_model, data_loader):
    filename = os.path.join(args.gen_dir, args.output_file)
    if not os.path.isdir(args.gen_dir):
        os.makedirs(args.gen_dir)
    generating(args, args.num_try, num_events, sim_model, data_loader, filename)


def save_model(model, optimizer, args):
    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.output_dir, args.save_model_name))

def load_model(model, optimizer, args):
    checkpoint = torch.load(os.path.join(args.output_dir, args.restore_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(args.device)
    model.eval()

################################# auxilliary NNs and classes #######################################

class ContextEmbedder(torch.nn.Module):
    """ Small NN to be used for the embedding of the conditionals """
    def __init__(self, input_size, output_size):
        """ input_size: length of context vector
            output_size: length of context vector to be fed to the flow
        """
        super(ContextEmbedder, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, (input_size+output_size)//2)
        self.layer2 = torch.nn.Linear((input_size+output_size)//2, (input_size+output_size)//2)
        self.output = torch.nn.Linear((input_size+output_size)//2, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.output(x)
        return out


@torch.no_grad()
def generating(args, num_pts, N_evt, sim_model, data_loader, filename):
    y_label = None
    dedx_ori = None
    isFirst = True
    for i, data in enumerate(data_loader):
        if isFirst:
            y_label = data['label']
            dedx_ori= data['dedx']
            isFirst = False
        else:
            dedx_ori = torch.cat((dedx_ori, data['dedx']), 0)
            y_label  = torch.cat((y_label , data['label']), 0)
        if dedx_ori.size()[0] >= N_evt:break
    dedx_ori = dedx_ori[0:N_evt]
    y_label  = y_label [0:N_evt]
    for i in range(0, y_label.size()[0], args.gen_batch):
        tmp_label   = y_label [i:i+args.gen_batch,:]
        tmp_dedx_ori= dedx_ori[i:i+args.gen_batch,:]
        dedx_dist_unit = sample_flow(sim_model, num_pts, args, tmp_label).to('cpu') if args.check_pt == False else jit_sample_flow(sim_model, args, tmp_label)
        dedx_dist = dedx_scale*dedx_dist_unit
        tmp_filename = filename.replace('.hdf5','_%d.hdf5'%( int(i/args.gen_batch) ) )
        print('save to %s'%tmp_filename, file=open(args.results_file, 'a'))
        #print('dedx_dist =', dedx_dist.size(),',tmp_dedx_ori=',tmp_dedx_ori.size(),',tmp_label=',tmp_label.size(),',tmp_filename=',tmp_filename)
        dedx_dist = torch.squeeze(dedx_dist,2)
        save_samples_to_file(dedx_sim=dedx_dist, dedx_ori=tmp_dedx_ori, real_label=tmp_label, filename=tmp_filename)


def train_flow(sim_model, train_data, test_data, optim, args):
    """ trains the flow that learns the distributions """
    best_eval_logprob_rec = float('-inf')

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], gamma=0.5)##py1.60
    if args.schedulerType == 1:
        peak_lr = 0.001
        warmup_iter = 5 if args.restore == False else -1
        eps = 1e-4
        sf = 0.7
        lr_schedule = LRWarmUPSF(optimizer=optim, warmup_iteration=warmup_iter, target_lr=peak_lr, threshold=eps, sf=sf)
        lr_schedule.step(1,999)
    if args.input_lr !=0:
        for param_group in optim.param_groups:
            param_group['lr'] = input_lr

    print(datetime.datetime.now(),file=open(args.results_file, 'a'))
    for epoch in range(args.num_epochs):
        sim_model.train()
        loglike_train = []
        tmp_lr = 0
        for param_group in optim.param_groups:
            tmp_lr = param_group['lr']
        for i, data in enumerate(train_data):
            x0 = data['dedx']
            x0 = x0.float()
            y = data['label'].to(args.device)
            #print(x0.type(),y.type())
            x = (x0/dedx_scale).clamp_(0., 1.).to(args.device)
            x = logit_trafo(x)
            #print(x.type(),y.type())
            loss = - sim_model.log_prob(x, y).mean(0)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loglike_train.append(loss.item())
        logprob_mean_train = np.mean(loglike_train)
        logprob_std_train = np.std(loglike_train)
        output = 'Flow: Training (epoch {}) -- '.format(epoch+1) + '-logp = {:.3f} +/- {:.3f}, lr is {:f}'
        print(output.format(logprob_mean_train, logprob_std_train, tmp_lr), file=open(args.results_file, 'a'))
        print(datetime.datetime.now(),file=open(args.results_file, 'a'))
        logprob_mean = float('-inf') 
        with torch.no_grad():
            sim_model.eval()
            loglike = []
            for data in test_data:
                x0 = data['dedx']
                x0 = x0.float()
                y = data['label'].to(args.device)
                x = (x0/dedx_scale).clamp_(0., 1.).to(args.device)
                x = logit_trafo(x)
                loglike.append(sim_model.log_prob(x, y))
            logprobs = torch.cat(loglike, dim=0).to(args.device)
            logprob_mean = logprobs.mean(0)
            logprob_std = logprobs.var(0).sqrt()
            output = 'Flow: Evaluate (epoch {}) -- '.format(epoch+1) + 'logp = {:.3f} +/- {:.3f}, lr is {:f}'
            print(output.format(logprob_mean, logprob_std, tmp_lr), file=open(args.results_file, 'a'))
        if args.schedulerType == 1:
            lr_schedule.step(epoch+2,logprob_mean_train)
        else:
            lr_schedule.step()
        if logprob_mean >= best_eval_logprob_rec:
            best_eval_logprob_rec = logprob_mean
            save_model(sim_model,optim, args)
    print(datetime.datetime.now(),file=open(args.results_file, 'a'))



@torch.no_grad()
def sample_flow(sim_model, num_pts, args, label_data=None):
    sim_model.eval()
    samples = sim_model.sample(num_pts, label_data.to(args.device))
    #samples = inverse_logit(samples.squeeze())
    samples = inverse_logit(samples)
    return samples


@torch.no_grad()
def save_flow(sim_model, args):
    sim_model.eval()
    tmp_device = torch.device("cpu")
    sim_model.to(tmp_device)
    #sim_model.double()##change to double 
    sim_model.float()##change to double 
    example_input = torch.tensor( [[0.0784, 0.4385, 0.8700, 0.3949]] )
    module = torch.jit.trace_module(sim_model, {'forward':example_input.float()})
    #module.double()##change to double 
    module.float()##change to double 
    module.save(args.pt_file_path)
    print('saved %s'%args.pt_file_path)

@torch.no_grad()
def jit_sample_flow(sim_model, args, label_data):
    sim_model.eval()
    tmp_device = torch.device("cpu")
    label_data = label_data.to(tmp_device)
    noise = torch.randn(label_data.size()[0], 1)
    example_input = torch.cat((label_data,noise), 1)
    samples = sim_model.forward(example_input)
    samples = inverse_logit(samples)##(1,1)
    return samples
####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()

    cond_label_size = 3#mom,theta,nhit
    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda)  if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))

    # get dataloaders
    train_dataloader, test_dataloader = get_dataloader(args.particle_type,
                                                       args.data_dir,
                                                       device=args.device,
                                                       batch_size=args.batch_size)
    dedx_scale = 1 
    if args.particle_type == 'p+' or args.particle_type == 'p-':
        dedx_scale = 22000 # for betagamma = 0.1 
    elif args.particle_type == 'pi+' or args.particle_type == 'pi-':
        dedx_scale = 2000
    elif args.particle_type == 'k+' or args.particle_type == 'k-':
        dedx_scale = 5000
    else:
        raise ValueError("Wrong dedx_scale")
    flow_params = {'num_blocks': args.num_block, #num of layers per block, default 2
                              'features': args.num_feature,
                              'context_features': 3, #1,
                              'hidden_features': args.hidden_features, #default is 64
                              'use_residual_blocks': False,
                              'use_batch_norm': False,
                              'dropout_probability': 0.,
                              'activation':getattr(F, args.activation_fn),
                              'random_mask': False,
                              'num_bins': 8,
                              'tails':'linear',
                              'tail_bound': 14,
                              'min_bin_width': 1e-6,
                              'min_bin_height': 1e-6,
                              'min_derivative': 1e-6}
    flow_blocks = []
    for _ in range(6):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params))
        flow_blocks.append(transforms.RandomPermutation(args.num_feature))
    flow_transform = transforms.CompositeTransform(flow_blocks)
    # _sample not implemented:
    flow_base_distribution = distributions.StandardNormal(shape=[args.num_feature])
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    sim_model = flow.to(args.device)
    sim_optimizer = torch.optim.Adam(sim_model.parameters(), lr=0.001)
    print(sim_model)
    print(sim_model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in sim_model.parameters() if p.requires_grad)

    print("setup has {} parameters".format(int(total_parameters)))
    print("setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))

    if args.training:
        print("do training:", file=open(args.results_file, 'a'))
        if args.restore:
            print("restoreing from %s"%args.restore_file, file=open(args.results_file, 'a'))
            load_model(sim_model, sim_optimizer, args)
        train_flow(sim_model, train_dataloader, test_dataloader, sim_optimizer, args)

    if args.generate_to_file:
        load_model(sim_model, sim_optimizer, args)
        if args.use_test_dataloader == False:
            generate_to_file(args, args.gen_events, sim_model=sim_model, data_loader=train_dataloader)
        else:
            generate_to_file(args, args.gen_events, sim_model=sim_model, data_loader=test_dataloader)
    if args.save_pt:
        load_model(sim_model, sim_optimizer, args)
        save_flow(sim_model, args)

    if args.check_pt:
        m_net =  torch.jit.load(args.check_pt_file)
        example_input = torch.tensor( [[0.0784, 0.4385, 0.8700, 0.3949]] )
        output = m_net.forward( example_input.float() )
        print('example_input=',example_input,',output=',output)
        #generate_to_file(args, num_events=1000000, sim_model=m_net, data_loader=train_dataloader)
        #generate_to_file(args, num_events=10, sim_model=m_net, data_loader=train_dataloader)

    if args.saveONNX:
        load_model(sim_model, sim_optimizer, args)
        sim_model.eval()
        tmp_device = torch.device("cpu")
        sim_model.to(tmp_device)
        sim_model.float()##change to double 
        example_input = torch.tensor( [[0.0784, 0.4385, 0.8700, 0.3949]] )
        #module = torch.jit.trace_module(sim_model, {'forward':example_input.float()})
        #module.float()##change to double 
        #print('saved %s'%args.pt_file_path)
        # Export the model
        torch.onnx.export(sim_model,               # model being run
                  example_input,                   # model input (or a tuple for multiple inputs)
                  args.onnx_file_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  opset_version=11,
                  verbose=True,
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #              'output' : {0 : 'batch_size'}}
        )
