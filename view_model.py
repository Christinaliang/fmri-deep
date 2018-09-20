import numpy as np
import torch
from matplotlib import pyplot as plt

from options import load_options

from os.path import join

name = 'markov_rest'
save = join('../models', name)

options = load_options(name)
train, test = options['dataset']
model = options['model']
optimizer = options['optimizer']

model.load_state_dict(torch.load(join(save, 'model.pth')))
losses = np.load(join(save, 'losses.npy'))

# mat = next(model.fc.parameters()).detach().numpy()
img = train[0][1]
# img_flat = img.reshape(img.shape[0], -1)
# corr = np.corrcoef(img_flat.T)

frame_0 = img[0]
print("Simulating...")
#img_sim = model.simulate(frame_0, img.shape[0])
#img_sim = np.transpose(img_sim, axes=(1,2,3,4,0))[0] # change axes for atlas
img_step = model.simulate_step(img)
img_step = np.transpose(img_step, axes=(1,2,3,4,0))[0]
#img_act = np.transpose(img.numpy(), axes=(1,2,3,4,0))[0]

#np.save(join(save, 'img_act.npy'), img_act)
#np.save(join(save, 'img_sim.npy'), img_sim)
np.save(join(save, 'img_step.npy'), img_step)