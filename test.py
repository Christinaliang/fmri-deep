import numpy as np
import torch
from torch.autograd import Variable

from options import load_options

from os import mkdir
from os.path import exists, join
import sys
import time

def main():
    if len(sys.argv) not in (2, 3):
        print('Arguments:')
        print('1: Name of experiment (must be defined in options.py)')
        print('2 (optional): Cuda device number')
        print('Example:')
        print('python3 test.py dncnn_mag 0')
        return
    
    # Parse cmd line arguments
    name = sys.argv[1]
    save = join('../models', name)
    if len(sys.argv) >= 3:
        device = int(sys.argv[2])
    else:
        device = None
    print(name, device)
    
    # Load cmd line arguments
    options = load_options(name)
    train, test = options['dataset']
    model = options['model']
    
    # Prepare saving, load model if resuming
    print('Loading model:', name)
    model.load_state_dict(torch.load(join(save, 'model.pth')))
    if device is not None:
        torch.cuda.set_device(device)
        model.cuda()
    
    # Train the model
    print('Generating sample...')
    start = time.time()
    sample = model.sample() # will be different for each model - update this
    np.save(join(save, 'sample.npy'), sample.cpu().detach().numpy())
    
    print('Time elapsed: %.3f' % (time.time() - start))
    
if __name__ == '__main__':
    main()