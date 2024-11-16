# retico-dopetrack
A ReTiCo module for pose-tracking, based on https://github.com/naver/dope.

### Installation and requirements

Please download [this model](http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz)
and place it in the `retico_dopetrack/src/models` directory.

Required Packages:  
- pytorch  
- opencv-python  
- pillow  
- numpy  
- scipy  
- matplotlib  

In addition, this module relies on [retico-core](https://github.com/retico-team/retico-core) 
and [retico-vision](https://github.com/retico-team/retico-vision).

See https://github.com/naver/dope for more information.

### Example
```python
import sys, os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

prefix = '/path/to/prefix/'
sys.path.append(prefix+'retico-core')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-dopetrack')

from retico_vision.vision import WebcamModule 
from retico_core.debug import DebugModule
from retico_dopetrack.dopetrack import DopeTrackingModule

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

### Citation
The source code this module is based on comes from https://github.com/naver/dope, which is 
licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. Minor 
changes have been made to facilitate better interfacing. The DOPE model is a result of the 
following [ECCV'20 paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710375.pdf):

```
@inproceedings{dope,
  title={{DOPE: Distillation Of Part Experts for whole-body 3D pose estimation in the wild}},
  author={{Weinzaepfel, Philippe and Br\'egier, Romain and Combaluzier, Hadrien and Leroy, Vincent and Rogez, Gr\'egory},
  booktitle={{ECCV}},
  year={2020}
}
```
