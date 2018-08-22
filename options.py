import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import data as d
import model_blocks as mb
import model_autoenc as ma
import model_transition as mt
import model_pixel as mp
import model_video as mv
import model_causal as mc
import transform as tm

def step_vanilla(model, optimizer, criterion, image, label):
    image = Variable(image).unsqueeze(0) # batch dimension
    label = Variable(label).unsqueeze(0)
            
    # Do one step
    optimizer.zero_grad()
    output = model(image)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    
    return loss.data.item()

def step_time(model, optimizer, criterion, image, label):
    image = Variable(image).unsqueeze(0) # batch dimension
                    
    # Do one time step at a time; time dimension should be first
    avg_loss = 0.0
    for t in range(label.shape[0] - 1):
        frame_in = label[t].unsqueeze(0) # current frame
        frame_out = label[t + 1].unsqueeze(0) # next frame
        
        optimizer.zero_grad()
        # use curr frame and static image to predict next frame
        output = model((frame_in, image))
        loss = criterion(output, frame_out)
        loss.backward()
        optimizer.step()
        
        avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
    return avg_loss

def step_time_l1(model, optimizer, criterion, image, label):
    image = Variable(image).unsqueeze(0) # batch dimension
                    
    # Do one time step at a time; time dimension should be first
    avg_loss = 0.0
    for t in range(label.shape[0] - 1):
        frame_in = label[t].unsqueeze(0) # current frame
        frame_out = label[t + 1].unsqueeze(0) # next frame
        
        optimizer.zero_grad()
        # use curr frame and static image to predict next frame
        output = model(frame_in)
        loss = criterion(output, frame_out, model)
        loss.backward()
        optimizer.step()
        
        avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
    return avg_loss

def step_pixel_recur(reset_every_t):
    def f(model, optimizer, criterion, image, label):
        image = Variable(image).unsqueeze(0) # batch dimension                
        # Do one time step at a time
        avg_loss = 0.0
        model.init_hidden_state() # subjects don't share states
        for t in range(label.shape[0] - 1):        
            # print(t)
            f_in = label[t].unsqueeze(0) # current frame
            f_out = label[t + 1].unsqueeze(0) # next frame
            
            optimizer.zero_grad()
            output = model((f_in, f_out, image))
            loss = criterion(output, f_out)
            
            if t % reset_every_t == reset_every_t - 1:
                # keep old state but remove graph to save memory.
                loss.backward()
                model.detach_hidden_state()
            else:
                loss.backward(retain_graph = True)
            optimizer.step()
            
            avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
        return avg_loss
    return f

def compute_loss_vanilla(dataset, criterion, model):
    """Given a model and dataset, computes average loss."""
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        output = model(image)
        avg += (criterion(output, label).data.item() - avg) / (i + 1)
    return avg

def compute_loss_time(dataset, criterion, model):
    """Given a model and dataset, computes average loss."""
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        
        image = Variable(image).unsqueeze(0) # batch dimension
        # Do one time step at a time
        avg_loss = 0.0
        for t in range(label.shape[0] - 1):
            frame_in = label[t].unsqueeze(0) # current frame
            frame_out = label[t + 1].unsqueeze(0) # next frame
            
            output = model((frame_in, image)) # use curr frame and static
            loss = criterion(output, frame_out) # to predict next frame
            
            avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
        avg += (avg_loss - avg) / (i + 1)
    return avg

def compute_loss_pixel_recur(dataset, criterion, model):
    """Given a model and dataset, computes average loss."""
    avg = 0.0
    for i, example in enumerate(dataset):
        image, label = example
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        
        image = Variable(image).unsqueeze(0) # batch dimension
        # Do one time step at a time
        avg_loss = 0.0
        model.init_hidden_state() # subjects don't share states
        for t in range(label.shape[0] - 1):
            f_in = label[t].unsqueeze(0) # current frame
            f_out = label[t + 1].unsqueeze(0) # next frame
            
            output = model((f_in, f_out, image))
            loss = criterion(output, f_out)
            
            avg_loss += ((loss.data.item() - avg_loss) / (t + 1))
        avg += (avg_loss - avg) / (i + 1)
    return avg

def no_test(d, c, m):
    return 0

def load_options(name):
    """Saves experiment options under names to load in train and test."""
    if name == 'rest_conn_l1':
        """Implement lambda grid search!"""
        tr1 = tm.Transforms((tm.ChannelDim(), tm.Decimate(),
                             tm.Transpose((4,0,1,2,3)),
                             tm.ToTensor()), apply_to = 'label')
        tr2 = tm.Transforms((tm.ToTensor(),), apply_to = 'image')
        tr = mb.MultiModule((tr1, tr2))
        train = d.SingleRestDataset('../data/train/NC01', tr)
        test = []
        model = mb.FC
        optimizer = optim.Adam
        l1 = 0.0001
        mse = nn.MSELoss()
        def loss_f(pred, act, net):
            l1_loss = mc.l1reg(model)
            error = mse(pred, act)
            # print('l1:', l1_loss, l1*l1_loss, 'mse:', error)
            return l1*l1_loss + error
        criterion = loss_f
        
        example = train[0][1]
        print(example.shape)
        ch, shape = example.shape[1], np.array(example.shape[2:])
        model = model(ch, shape, ch, shape)
        
        step, compute_loss = step_time_l1, no_test
    if name == 'rest_conn_nol1':
        """Implement lambda grid search!"""
        tr1 = tm.Transforms((tm.ChannelDim(), tm.Decimate(),
                             tm.Transpose((4,0,1,2,3)),
                             tm.ToTensor()), apply_to = 'label')
        tr2 = tm.Transforms((tm.ToTensor(),), apply_to = 'image')
        tr = mb.MultiModule((tr1, tr2))
        train = d.SingleRestDataset('../data/train/NC01', tr)
        test = []
        model = mb.FC
        optimizer = optim.Adam
        l1 = 0
        mse = nn.MSELoss()
        def loss_f(pred, act, net):
            l1_loss = mc.l1reg(model)
            error = mse(pred, act)
            # print('l1:', l1_loss, l1*l1_loss, 'mse:', error)
            return l1*l1_loss + error
        criterion = loss_f
        
        example = train[0][1]
        print(example.shape)
        ch, shape = example.shape[1], np.array(example.shape[2:])
        model = model(ch, shape, ch, shape)
        
        step, compute_loss = step_time_l1, no_test
    if name == 't2_vae':
        tr = tm.Transforms((tm.ChannelDim(), tm.Decimate(), tm.Normalize(), 
                            tm.ToTensor()))
        train = d.T2AutoencDataset('../data/train', tr)
        test = d.T2AutoencDataset('../data/test', tr) # retest: tested on train
        model = ma.BrainVAE
        optimizer = optim.Adam
        criterion = ma.loss_VAE
        
        example = train[0][0] # patient 0 = (image, label)
        ch, shape = example.shape[0], np.array(example.shape[1:])
        model = model(ch, shape)
        
        step, compute_loss = step_vanilla, compute_loss_vanilla
    if name == 't2_autoenc':
        tr = tm.Transforms((tm.Normalize(), tm.Decimate(), tm.ChannelDim(), 
                            tm.ToTensor()))
        train = d.T2AutoencDataset('../data/train', tr)
        test = d.T2AutoencDataset('../data/test', tr)
        model = ma.BrainAutoenc
        optimizer = optim.Adam
        criterion = nn.MSELoss()
        
        example = train[0][0]
        ch, shape = example.shape[0], np.array(example.shape[1:])
        model = model(ch, shape)
        
        step, compute_loss = step_vanilla, compute_loss_vanilla
    if name == 'dti_rest_trans': # train with train_recurrent
        tr_dti = tm.Transforms((tm.Transpose((3,0,1,2)), tm.Decimate()),
                               apply_to = 'image')
        tr_rest = tm.Transforms((tm.ChannelDim(), tm.Transpose((4,0,1,2,3))), 
                                apply_to = 'label')
        tr_both = tm.Transforms((tm.Normalize(), tm.ToTensor()))
        tr = mb.MultiModule((tr_dti, tr_rest, tr_both))
        # DTI: 256 x 256 x 67 x 6: normalize, transpose(3,0,1,2), decimate
        # Rest: 64 x 64 x 38 x 205: normalize, channeldim, transpose(4,0,1,2,3)
        # splitting of time points is done in train_recurrent.
        # DTI: 6 x 256 x 256 x 67
        # Rest: 205 x 1 x 64 x 64 x 38
        train = d.DTIRestDataset('../data/train', tr)
        test = d.DTIRestDataset('../data/test', tr)
        
        image, label = train[0]
        s_ch, s_shape = image.shape[0], np.array(image.shape[1:])
        i_ch, i_shape = label.shape[1], np.array(label.shape[2:])
        model = mt.WeightTransitionNet(i_ch, i_shape, s_ch, s_shape, i_ch)
        
        optimizer = optim.Adam
        criterion = nn.MSELoss() # change to pixel loss later
        
        step, compute_loss = step_time, compute_loss_time
    if name == 'dti_rest_video':
        tr_dti = tm.Transforms((tm.Transpose((3,0,1,2)), tm.Decimate()),
                               apply_to = 'image')
        tr_rest = tm.Transforms((tm.ChannelDim(), tm.Transpose((4,0,1,2,3))), 
                                apply_to = 'label')
        tr_both = tm.Transforms((tm.Normalize(), tm.ToTensor()))
        tr = mb.MultiModule((tr_dti, tr_rest, tr_both))
        # DTI: 256 x 256 x 67 x 6: normalize, transpose(3,0,1,2), decimate
        # Rest: 64 x 64 x 38 x 205: normalize, channeldim, transpose(4,0,1,2,3)
        # splitting of time points is done in train_recurrent.
        # DTI: 6 x 256 x 256 x 67
        # Rest: 205 x 1 x 64 x 64 x 38
        train = d.DTIRestDataset('../data/train', tr)
        test = d.DTIRestDataset('../data/test', tr)
        
        image, label = train[0]
        s_ch, s_shape = image.shape[0], np.array(image.shape[1:])
        f_ch, f_shape = label.shape[1], np.array(label.shape[2:])
        model = mv.WeightPixelNet(f_ch, f_shape, s_ch, s_shape, f_ch * 4)
        
        optimizer = optim.Adam
        criterion = mp.logistic_mixture_loss
        
        step, compute_loss = step_pixel_recur(8), compute_loss_pixel_recur
    if name == 'rest_vpn':
        tr_rest = tm.Transforms((tm.ChannelDim(), tm.Transpose((4,0,1,2,3))), 
                                apply_to = 'label')
        tr_both = tm.Transforms((tm.Normalize(), tm.ToTensor()))
        tr = mb.MultiModule((tr_rest, tr_both))
        train = d.RestDataset('../data/train', tr)
        test = d.RestDataset('../data/test', tr)
        
        image, label = train[0]
        ch, shape = label.shape[1], np.array(label.shape[2:])
        model = mv.ResPixelNet(ch, shape, 8)
        
        optimizer = optim.Adam
        criterion = mp.logistic_mixture_loss
        
        step, compute_loss = step_pixel_recur(5), compute_loss_pixel_recur
    if name == 'dti_rest_gate':
        tr_dti = tm.Transforms((tm.Transpose((3,0,1,2)), tm.Decimate()),
                               apply_to = 'image')
        tr_rest = tm.Transforms((tm.ChannelDim(), tm.Transpose((4,0,1,2,3))), 
                                apply_to = 'label')
        tr_both = tm.Transforms((tm.Normalize(), tm.ToTensor()))
        tr = mb.MultiModule((tr_dti, tr_rest, tr_both))
        # DTI: 256 x 256 x 67 x 6: normalize, transpose(3,0,1,2), decimate
        # Rest: 64 x 64 x 38 x 205: normalize, channeldim, transpose(4,0,1,2,3)
        # splitting of time points is done in train_recurrent.
        # DTI: 6 x 256 x 256 x 67
        # Rest: 205 x 1 x 64 x 64 x 38
        train = d.DTIRestDataset('../data/train', tr)
        test = d.DTIRestDataset('../data/test', tr)
        
        image, label = train[0]
        x_ch, x_shape = image.shape[0], np.array(image.shape[1:])
        f_ch, f_shape = label.shape[1], np.array(label.shape[2:])
        model = mv.GatePixelNet(f_ch, f_shape, x_ch, x_shape, 8)
        
        optimizer = optim.Adam
        criterion = mp.logistic_mixture_loss
        
        step, compute_loss = step_pixel_recur(2), compute_loss_pixel_recur
    optimizer = optimizer(model.parameters())
    return {'dataset': (train, test), 'model': model, 'optimizer': optimizer, 
            'criterion': criterion, 'funcs': (step, compute_loss)}