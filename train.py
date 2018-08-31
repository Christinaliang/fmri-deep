import numpy as np
import torch
from torch.autograd import Variable

from options import load_options
from functions import Logger, to_boolean

from os import mkdir
from os.path import exists, join
import sys
import time

def save_state(obj, file):
    state = obj.state_dict()
    cpu_state = {}
    for key in state:
        try:
            cpu_state[key] = state[key].cpu()
        except AttributeError:
            cpu_state[key] = state[key]
    torch.save(cpu_state, file)

def main():
    if len(sys.argv) not in (4, 5):
        print('Arguments:')
        print('1: Name of experiment (must be defined in options.py)')
        print('2: Num epochs')
        print('3: Load existing?')
        print('4 (optional): Cuda device number')
        print('Example:')
        print('python3 train.py dncnn_mag 3 False 0')
        return
    
    # Parse cmd line arguments
    name = sys.argv[1]
    save = join('../models', name)
    epochs = int(sys.argv[2])
    load = to_boolean(sys.argv[3])
    if len(sys.argv) >= 5:
        device = int(sys.argv[4])
    else:
        device = None
    print(name, save, epochs, load, device)
    
    # Load cmd line arguments
    options = load_options(name)
    train, test = options['dataset']
    model = options['model']
    optimizer = options['optimizer']
    
    # Prepare saving, load model if resuming
    sys.stdout = Logger(join(save, 'log.txt'))
    losses = None
    if not exists(save):
        mkdir(save)
    elif load:
        print('Loading model:', name)
        model.load_state_dict(torch.load(join(save, 'model.pth')))
        optimizer.load_state_dict(torch.load(join(save, 'optimizer.pth')))
        losses = np.load(join(save, 'losses.npy'))
    if device is not None:
        torch.cuda.set_device(device)
        model.cuda()
        
    # Train the model
    print('Beginning training...')
    print('Name:', name)
    print('Epochs:', epochs)
    print('Examples per epoch:', len(train))
    start = time.time()
    for e in range(epochs):
        train_loss, test_loss = 0.0, 0.0
        
        model.train()
        train.shuffle()
        for i, example in enumerate(train):
            # Load example
            image, label = example
            image = Variable(image).unsqueeze(0)
            label = Variable(label).unsqueeze(0)
            
            step_loss = model.step(image, label, optimizer = optimizer)
            # Compute average so far
            train_loss += ((step_loss - train_loss) / (i + 1))
            
        # Test, log, and save once every epoch
        model.eval()
        for i, example in enumerate(test):
            image, label = example
            image = Variable(image).unsqueeze(0)
            label = Variable(label).unsqueeze(0)
            
            step_loss = model.step(image, label)
            test_loss += ((step_loss - test_loss) / (i + 1))
            
        if losses is None:
            losses = np.array([train_loss, test_loss])
        else:
            losses = np.vstack((losses, [train_loss, test_loss]))
        print('[%d] Train loss: %.3f, Test loss: %.3f, Time elapsed: %.3f'
              % (e + 1, train_loss, test_loss, time.time() - start))
        save_state(model, join(save, 'model.pth'))
        save_state(optimizer, join(save, 'optimizer.pth'))
        np.save(join(save, 'losses.npy'), losses)
    
    print('Finished %d epochs.' % (epochs))
    print('Time elapsed: %.3f' % (time.time() - start))
    
if __name__ == '__main__':
    main()