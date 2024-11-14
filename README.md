# retico-dope
A ReTiCo module for DOPE pose-tracking. See below for more information.

### Installation and requirements

See https://github.com/naver/dope 

Required Packages:  
- pytorch  
- torchvision
- opencv
- numpy  


### Example
```python
import sys, os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

prefix = '/path/to/prefix/'
sys.path.append(prefix+'retico-core')
sys.path.append(prefix+'retico-vision')

from retico_vision.vision import WebcamModule 
from retico_core.debug import DebugModule
from dopetrack import DopeTrackingModule

cam = WebcamModule()
dope = DopeTrackingModule(visualizer=True)
debug = DebugModule(print_payload_only=True)

cam.subscribe(dope)
dope.subscribe(debug)

cam.run()
dope.run()
debug.run()

input()

debug.stop()
dope.stop()
cam.stop()
```

Citation
```
@inproceedings{dope,
  title={{DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild}},
  author={{Weinzaepfel, Philippe and Br\'egier, Romain and Combaluzier, Hadrien and Leroy, Vincent and Rogez, Gr\'egory},
  booktitle={{ECCV}},
  year={2020}
}
```
