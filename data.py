""" Dataloader for calorimeter data.
    Inspired by https://github.com/kamenbliznashki/normalizing_flows

    Used for
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import os
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

ALPHA = 1e-6
def logit(x):
    return np.log(x / (1.0 - x))

def logit_trafo(x):
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)


class dEdxDataset(Dataset):
    """dataset of dE/dx"""

    def __init__(self, path_to_file, particle_type, prefix):
        """
        Args:
            path_to_file (string): path to folder of .hdf5 files
            particle_type (string): name of particle: p+, p- , k+, k-, pi+, pi-
        """
        assert particle_type in ['p+','p-','k+','k-','pi+','pi-','e+','e-','mu+','mu-']
        files = os.listdir(path_to_file)
        input_list = []

        for i in files:
            if prefix in i and particle_type in i:
                input_list.append('%s/%s'%(path_to_file,i))
        print('used data:', input_list)
        self.scale_back={}
        self.scale_back['p']={'mom':[0.6, 0.25],'theta':[90.0, 38.0],'nhit':[24.0, 6.0],'dedx':[1317.0, 798.0]}
        self.scale_back['k']={'mom':[0.0, 1.],'theta':[0.0, 1.0],'nhit':[0.0, 1.0],'dedx':[-300.0, 1.0]}##shift -300
        scale_back = None
        if particle_type == 'p+' or particle_type=='p-':
            scale_back = self.scale_back['p']
        elif particle_type == 'k+' or particle_type=='k-':
            scale_back = self.scale_back['k']
        if scale_back != None:
            print('scale back dedx=',scale_back['dedx'][1], scale_back['dedx'][0])  ##normalized: mom, theta, nhit, dEdx, mom_phy
        

        self.file_dedx  = None
        self.file_label = None
        isFirst = True
        for i in input_list:
            full_file = h5py.File(i, 'r')
            df = full_file['dataset'][:]
            #df = df [ np.logical_and(df[:,0] > 0, df[:,0] < 0.6), : ]##FIXME for k
            dedx        = np.float32( df[:,3:4] ) if scale_back == None else np.float64( df[:,3:4] )*scale_back['dedx'][1] + scale_back['dedx'][0]  ##normalized: mom, theta, nhit, dEdx, mom_phy
            label_mom   = np.float32( df[:,0:1] ) if scale_back == None else np.float64( df[:,0:1] )*scale_back['mom' ][1] + scale_back['mom' ][0]  ##normalized: mom, theta, nhit, dEdx, mom_phy
            label_theta = np.float32( df[:,1:2] ) if scale_back == None else np.float64( df[:,1:2] )*scale_back['theta' ][1] + scale_back['theta' ][0]  ##normalized: mom, theta, nhit, dEdx, mom_phy
            label_nhit  = np.float32( df[:,2:3] ) if scale_back == None else np.float64( df[:,2:3] )*scale_back['nhit' ][1] + scale_back['nhit' ][0]  ##normalized: mom, theta, nhit, dEdx, mom_phy
            #label_mom = np.log(label_mom)##FIXME for k
            labels      = np.float32( np.concatenate((label_mom, label_theta, label_nhit), axis=1)  )
            if isFirst:
                self.file_dedx  = dedx
                self.file_label = labels
                isFirst = False
            else:
                self.file_dedx  = np.float32( np.concatenate((self.file_dedx , dedx  ), axis=0)  )
                self.file_label = np.float32( np.concatenate((self.file_label, labels), axis=0)  )
            full_file.close()
        print('input data size=',self.file_dedx.shape)

    def __len__(self):
        # assuming file was written correctly
        return len(self.file_dedx)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'dedx': self.file_dedx[idx]}
        sample['label'] = self.file_label[idx]

        return sample

class fittedDataset(Dataset):
    """dataset of dE/dx"""

    def __init__(self, path_to_file, particle_type):
        """
        Args:
            path_to_file (string): path to folder of .hdf5 files
            particle_type (string): name of particle: p+, p- , k+, k-, pi+, pi-
        """
        #files = os.listdir(path_to_file)
        
        input_list = []
        with open(path_to_file,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n','')
                if '#' in line:continue
                input_list.append(line)
        #for i in files:
        #    if 'hdf5' in i:
        #        input_list.append('%s/%s'%(path_to_file,i))
        print('used data:', input_list)
        #self.scale_back={}
        #self.scale_back['p']={'mom':[0.3, 0.07],'theta':[90.0, 35.0],'nhit':[24.0, 6.0],'dedx':[3922.0, 1623.0]}
        #scale_back = None
        #if particle_type == 'p+' or particle_type=='p-':
        #    scale_back = self.scale_back['p']
        #
        #if scale_back != None:
        #    print(scale_back['dedx'][1], scale_back['dedx'][0])  ##normalized: mom, theta, nhit, dEdx, mom_phy
        

        self.file_dedx  = None
        self.file_mean  = None
        self.file_sigma  = None
        self.file_label = None
        isFirst = True
        for i in input_list:
            full_file = h5py.File(i, 'r')
            df = full_file['Fit'][:]
            df = df [ np.logical_and(df[:,4] > 0, df[:,6] > 0), : ]

            dedx        = df[:,3:4]/550. ##mom, theta, nhit, obs_dEdx, fit_mean, fit_mean_err, fit_sigma, fit_sigma_err, h_means, h_rms
            label_mom   = df[:,0:1]
            label_theta = df[:,1:2]
            label_nhit  = df[:,2:3]
            labels      = np.concatenate((label_mom, label_theta, label_nhit), axis=1)
            #labels[:,0] =  (labels[:,0]-0.3)/0.07
            #labels[:,1] =  (labels[:,1]-90.0)/35.
            #labels[:,2] =  (labels[:,2]-24)/6.
            fit_mean    = df[:,4:5]/550.
            fit_sigma   = df[:,6:7]/550.
            if isFirst:
                self.file_dedx   = dedx
                self.file_mean   = fit_mean
                self.file_sigma  = fit_sigma
                self.file_label = labels
                isFirst = False
            else:
                self.file_dedx  = np.float32( np.concatenate((self.file_dedx  , dedx       ), axis=0)  )
                self.file_mean  = np.float32( np.concatenate((self.file_mean  , fit_mean   ), axis=0)  )
                self.file_sigma = np.float32( np.concatenate((self.file_sigma , fit_sigma  ), axis=0)  )
                self.file_label = np.float32( np.concatenate((self.file_label , labels     ), axis=0)  )
            full_file.close()
        print('input label size=',self.file_label.shape)

    def __len__(self):
        # assuming file was written correctly
        return len(self.file_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'mean': self.file_mean[idx]}
        sample['sigma'] = self.file_sigma[idx]
        sample['label'] = self.file_label[idx]
        sample['dedx'] = self.file_dedx[idx]

        return sample


def get_dataloader(particle_type, data_dir, device, batch_size):

    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type is 'cuda' else {'pin_memory': False}

    train_dataset = dEdxDataset(data_dir, particle_type, 'train')
    test_dataset =  dEdxDataset(data_dir, particle_type, 'test' )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False, **kwargs)
    return train_dataloader, test_dataloader

def get_dataloader_fitted(particle_type, datafile_train, datafile_test, device, batch_size):

    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type is 'cuda' else {}

    train_dataset = fittedDataset(datafile_train, particle_type)
    test_dataset =  fittedDataset(datafile_test , particle_type)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,  shuffle=False, **kwargs)
    return train_dataloader, test_dataloader

def add_noise(input_tensor):
    noise = np.random.rand(*input_tensor.shape)*1e-8
    return input_tensor+noise


def save_samples_to_file(dedx_sim, dedx_ori, real_label, filename):
    dedx_sim = dedx_sim.to('cpu').numpy()
    dedx_ori = dedx_ori.to('cpu').numpy()
    real_label = real_label.to('cpu').numpy()
    df = np.concatenate((real_label, dedx_ori, dedx_sim), axis=1)
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('Pred', data=df)
    save_file.close()

def save_calib_to_file(real_label, dedx_ori, outputs, filename):
    real_label = real_label.to('cpu').numpy()
    dedx_ori = dedx_ori.to('cpu').numpy()
    outputs  = outputs.to('cpu').numpy()
    df = np.concatenate((real_label, dedx_ori, outputs), axis=1)
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('Pred', data=df)
    save_file.close()

def save_dedx_to_file(real_label, outputs, filename):
    real_label = real_label.to('cpu').numpy()
    outputs  = outputs.to('cpu').numpy()
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('label', data=real_label)
    save_file.create_dataset('Pred' , data=outputs)
    save_file.close()

def save_calib_to_file2(real_label, dedx, target1, target2, output1, output2, filename):
    real_label = real_label.to('cpu').numpy()
    dedx    = dedx.to('cpu').numpy()
    target1 = target1.to('cpu').numpy()
    target2 = target2.to('cpu').numpy()
    output1  = output1.to('cpu').numpy()
    output2  = output2.to('cpu').numpy()
    df = np.concatenate((real_label, dedx, target1, target2, output1, output2), axis=1)
    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('Pred', data=df)
    save_file.close()

