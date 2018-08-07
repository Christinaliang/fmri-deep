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
        print('python3 test.py dncnn_smallm_mag 0')
        exit()
    
    name = sys.argv[1]
    device = int(len(sys.argv) >= 3 and sys.argv[2])
    save = join('../models', name)
    
    options = load_options(name)
    train, test = options['dataset']
    model = options['model']
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        map_location = None
        model.cuda()
    else:
        map_location = lambda storage, loc: storage
        
    model.load_state_dict(torch.load(join(save, 'model.pth'), 
                                     map_location = map_location))
    
    model.eval()
    
    print('Generating test predictions...')
    start = time.time()
    for i, example in enumerate(test):
        # Load example
        image, label = example
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        image = Variable(image).unsqueeze(0) # batch dimension
        label = Variable(label).unsqueeze(0)
        
        output = model(image)
        
        ex_dir = join(save, 'ex' + str(i))
        if not exists(ex_dir):
            mkdir(ex_dir)
        print(ex_dir)
        if type(output) is dict:
            for k in output.keys():
                np.save(join(ex_dir, k + '.npy'), 
                        torch.squeeze(output[k]).cpu().data.numpy())
        else:
            np.save(join(ex_dir, 'pred.npy'),
                    torch.squeeze(output).cpu().data.numpy())
        np.save(join(ex_dir, 'image.npy'), 
                torch.squeeze(image).cpu().data.numpy())
        np.save(join(ex_dir, 'label.npy'), 
                torch.squeeze(label).cpu().data.numpy())
    print('Time elapsed: %.3f' % (time.time() - start))

if __name__ == '__main__':
    main()