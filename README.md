# peaknet4antfarm
A peaknet API with pytorch backbone


## Example

```
peaknet = Peaknet() # Init a Peaknet instance
peaknet.loadDNWeights() # Load newpeaksv9 network and pretrained weights 
peaknet.model.cuda() # transfer the net to GPU
peaknet.model.print_network() # print network 
next(peaknet.model.parameters()) # a good way to see if the net is on GPU
```

## API

### train 
```
peaknet.train( imgs, labels, box_size = 7 )
```

`imgs` is a numpy array with dimensions `(n,m,h,w)`. `imgs` will be treated as a stack of `n`x`m` tiles.
`labels` is a list of tutple of length `n`. Each item in the list is a tutple of three numpy arrays `s`, `r`, `c`, where `s` is an array of integers 0~`(m-1)`.

### model access
```
peaknet.model
```
returns the current model

### update model
```
peaknet.updateModel( newModel )
```
replaces the current model with `newModel`, including the network and the weights

### update grad
```
peaknet.updateModel( newModel )
```
replaces gradients in the current model with that from `newModel`. `newModel` must have same network as current model.

### optimize
```
peaknet.optimize()
```
performs one step of SGD optimization.
