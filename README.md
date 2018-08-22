# peaknet4antfarm
A peaknet API with pytorch backbone


## Example

```
from peaknet import Peaknet
peaknet = Peaknet() # Init a Peaknet instance
peaknet.loadDNWeights() # Load newpeaksv9 network and pretrained weights 
peaknet.model.cuda() # transfer the net to GPU
peaknet.model.print_network() # print network 
next(peaknet.model.parameters()) # a good way to see if the net is on GPU
```

## Setup

Add the following lines to your `~/.bashrc` file.
```
# PEAKNET4ANTFARM
export PYTHONPATH=/reg/neh/home5/liponan/ai/peaknet4antfarm:/reg/neh/home5/liponan/ai/pytorch-yolo2:$PYTHONPATH
```

I recommend creating an isolated environment for running Peaknet, as it requires an old version of PyTorch.
```
conda create --name antfarm python=2.7 pytorch=0.1.12 torchvision numpy h5py
conda activate antfarm
conda install --channel lcls-rhel7 psana-conda
```


## API

### train (for client)
```
peaknet.train( imgs, labels, box_size = 7 )
```

`imgs` is a numpy array with dimensions `(n,m,h,w)`. `imgs` will be treated as a stack of `n`x`m` tiles.
`labels` is a list of tutple of length `n`. Each item in the list is a tutple of three numpy arrays `s`, `r`, `c`, where `s` is an array of integers 0~`(m-1)`.

### model access (for client)
```
peaknet.model
```
returns the current model

### update model (for client)
```
peaknet.updateModel( newModel )
```
replaces the current model with `newModel`, including the network and the weights

### update grad (for Queen)
```
peaknet.updateGrad( newModel )
```
replaces gradients in the current model with that from `newModel`. `newModel` must have same network as current model.

### optimize (for Queen)
```
peaknet.optimize()
```
performs one step of SGD optimization.
