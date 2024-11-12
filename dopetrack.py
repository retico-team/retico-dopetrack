import time
import sys
from os import path
import cv2
import torch.cuda
from torchvision.transforms.v2 import ToTensor
import numpy as np
import threading
from collections import deque

import retico_core
from retico_vision.vision import PosePositionsIU, ImageIU

from src import postprocess
from src import visu
from src.model import dope_resnet50, num_joints


class DopeTrackingModule(retico_core.AbstractModule):
    """
    A pose tracking module using a Distillation of Part Experts (DOPE) model.
    """

    @staticmethod
    def name():
        return "DOPE Posetracking"

    @staticmethod
    def description():
        return "A pose tracking module using Distillation of Part Experts (DOPE)"

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return PosePositionsIU

    def __init__(self, visualizer=False, **kwargs):
        """
        :param visualizer: boolean to determine if webcam output with annotations is shown
        """

        super().__init__(**kwargs)
        self._run_tracker_active = False
        self.visualizer = visualizer
        self.queue = deque(maxlen=1)
        self._thisdir = path.realpath(path.dirname(__file__))

    def detect_pose(self, image, modelname='DOPErealtime_v1_0_0', postprocessing='ppi', half=True):
        """
        Perform pose detection on an image.

        :param image: Image to perform detections on.
        :param modelname: Name of model checkpoint to use. Defaults to 'DOPErealtime_v1_0_0'.
        :param postprocessing: The type of detection postprocessing to use (can be 'ppi' or 'nms'). Defaults to 'ppi'.
        :param half: Boolean indicating whether to use half computation or not. Defaults to True.
        """

        if postprocessing=='ppi':
            sys.path.append(self._thisdir + '/lcrnet-v2-improved-ppi')
            try:
                from src.lcr_net_ppi_improved import LCRNet_PPI_improved
            except ModuleNotFoundError:
                print("Failed to import PPI module. Using built-in NMS instead.")

        # use GPU if available, else use CPU
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # load model
        ckpt_fname = path.join(self._thisdir, 'src/models', modelname + '.pth.tgz')
        if not path.isfile(ckpt_fname):
            raise Exception(
                f'{ckpt_fname} does not exist, please download the model first and place it in the models/ folder'
            )

        print('Loading model', modelname)
        ckpt = torch.load(ckpt_fname, map_location=device)

        # Half computation is true by default. Change to false in case your device cannot handle half computation
        # Sets half computation usage
        ckpt['half'] = half
        ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000

        model = dope_resnet50(**ckpt['dope_kwargs'])
        if ckpt['half']: model = model.half()
        model = model.eval()
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)

        # load image
        print('Loading image')
        imlist = [ToTensor()(image).to(device)]
        if ckpt['half']: imlist = [im.half() for im in imlist]
        resolution = imlist[0].size()[-2:]

        # forward pass
        print('Running DOPE')
        with torch.no_grad():
            results = model(imlist, None)[0]

        # postprocess results (pose proposals integration, wrists/head assignment)
        print('Postprocessing')
        assert postprocessing in ['nms', 'ppi']
        parts = ['body', 'hand', 'face']
        detections = {}
        if postprocessing=='ppi':
            res = {k: v.float().data.cpu().numpy() for k,v in results.items()}
            for part in parts:
                detections[part] = LCRNet_PPI_improved(
                    res[part + '_scores'],
                    res['boxes'],
                    res[part + '_pose2d'],
                    res[part + '_pose3d'],
                    resolution,
                    **ckpt[part + '_ppi_kwargs']
                )
        else:
            for part in parts:
                dets, indices, bestcls = postprocess.DOPE_NMS(
                    results[part + '_scores'],
                    results['boxes'],
                    results[part + '_pose2d'],
                    results[part + '_pose3d'],
                    min_score=0.3
                )
                dets = {k: v.float().data.cpu().numpy() for k,v in dets.items()}
                detections[part] = [
                    {
                    'score': dets['score'][i],
                    'pose2d': dets['pose2d'][i,...],
                    'pose3d': dets['pose3d'][i,...]
                    } for i in range(dets['score'].size)
                ]
                if part == 'hand':
                    for i in range(len(detections[part])):
                        detections[part][i]['hand_isright'] = bestcls < ckpt['hand_ppi_kwargs']['K']

        return postprocess.assign_hands_and_head_to_body(detections)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)

    def _run_tracker(self):
        while self._run_tracker_active:
            if len(self.queue) == 0:
                time.sleep(0.05)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.payload
            frame = np.asarray(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Flip frame horizontally for natural (selfie) visualization
            frame = cv2.flip(frame, 1)

            # Get width, height of frame
            frame_height, frame_width, _ = frame.shape

            # Resize frame while maintaining aspect ratio
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Perform pose detection and assignment of hands and head to body
            detections, body_with_wrists, body_with_head = self.detect_pose(frame)

            # Display frame
            if self.visualizer:
                print('Displaying results')
                det_poses2d = {
                    part: np.stack([d['pose2d'] for d in part_detections], axis=0) if
                          len(part_detections) > 0 else
                          np.empty((0, num_joints[part], 2), dtype=np.float32) for
                          part, part_detections in detections.items()
                }
                scores = {
                    part: [d['score'] for d in part_detections] for
                          part, part_detections in detections.items()
                }
                output_image = visu.visualize_bodyhandface2d(
                    np.asarray(frame)[:, :, ::-1],
                    det_poses2d,
                    dict_scores=scores
                )
                cv2.imshow("Pose Detection", output_image)

                # Wait for key press and retrieve ASCII code
                k = cv2.waitKey(1) & 0xFF

                # Check if key was 'ESC'
                if k == 27:
                    break

            output_iu = self.create_iu(input_iu)
            output_iu.set_landmarks(frame, detections, None)
            self.append(retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD))

    def setup(self):
        self._run_tracker_active = True
        t = threading.Thread(target=self._run_tracker)
        t.start()

    def shutdown(self):
        self._run_tracker_active = False