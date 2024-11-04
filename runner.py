import os, sys

prefix = '/home/prigby/Retico/'
os.environ['RETICO'] = prefix + 'retico-core'
os.environ['RETICOV'] = prefix + 'retico-vision'

sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['RETICOV'])

from retico_vision.vision import WebcamModule
from dopetrack import DopeTrackingModule
from retico_core.debug import DebugModule

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
