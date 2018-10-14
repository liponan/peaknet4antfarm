import numpy as np
import sys

batch_size = 32

if len(sys.argv) < 2:
    raise Exception('no weights file provided')
if len(sys.argv) > 2:
    batch_size = int( sys.argv[2] ) 

zero = np.zeros((1,), dtype=np.int32)
f = open(sys.argv[1], 'rb')
header = np.fromfile(f, dtype=np.int32, count = 4 )
print(header)
rest = np.fromfile(f, dtype=np.float)
f.close()
g = open(sys.argv[1]+".fixed", "wb")
header.tofile(g)
zero.tofile(g)
rest.tofile(g)
g.close()
