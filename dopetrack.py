import time
import sys
from os import path
import cv2
import torch.cuda
from torchvision.transforms.v2 import ToTensor
import numpy as np
import threading
from collections import deque

from src import postprocess, visu3d
from src.dope import visu
from src.model import dope_resnet50, num_joints

import retico_core
from retico_vision.vision import PosePositionsIU, ImageIU


class DopeTrackingModule(retico_core.AbstractModule):
    """
    A pose tracking module using Distillation of Part Experts (DOPE)
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

    def __init__(self, visualizer=False, visualizer3D=False, **kwargs):
        """

        :param visualizer: boolean to determine if webcam output with annotations is shown
        :param kwargs:
        """

        super().__init__(**kwargs)
        self.visualizer = visualizer
        self.visualizer3D = visualizer3D

        self.queue = deque(maxlen=1)

        self._thisdir = path.realpath(path.dirname(__file__))


    def detect_pose(self, image, modelname='DOPErealtime_v1_0_0', postprocessing='ppi'):
        """
        Perform pose detection on an image.
        """

        if postprocessing=='ppi':
            sys.path.append(self._thisdir + '/lcrnet-v2-improved-ppi')
            try:
                from src.lcr_net_ppi_improved import LCRNet_PPI_improved
            except ModuleNotFoundError:
                raise Exception(
                    'To use the pose proposals integration (ppi) as postprocessing, please follow the '
                    'readme instruction by cloning our modified version of LCRNet_v2.0 here. Alternatively, '
                    'you can use --postprocess nms without any installation, with a slight decrease of performance.'
                )

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # load model
        ckpt_fname = path.join(self._thisdir, 'src/models', modelname + '.pth.tgz')
        if not path.isfile(ckpt_fname):
            raise Exception(
                f'{ckpt_fname} does not exist, please download the model first and place it in the models/ folder'
            )
        print('Loading model', modelname)
        ckpt = torch.load(ckpt_fname, map_location=device)
        # ckpt['half'] = False # uncomment this line in case your device cannot handle half computation
        ckpt['dope_kwargs']['rpn_post_nms_top_n_test'] = 1000
        model = dope_resnet50(**ckpt['dope_kwargs'])
        if ckpt['half']: model = model.half()
        model = model.eval()
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)

        # load image
        print('Loading image', image)
        imlist = [ToTensor()(image).to(device)]
        if ckpt['half']: imlist = [im.half() for im in imlist]
        resolution = imlist[0].size()[-2:]

        #forward pass
        print('Running DOPE')
        with torch.no_grad():
            results = model(imlist, None)[0]

        # postporcess results (pose proposals integration, wrists/head assignment
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

        # Assignment of hands and head to body
        detections, body_with_wrists, body_with_head = postprocess.assign_hands_and_head_to_body(detections)

        if self.visualizer:
            print('Displaying results')
            det_poses2d = {part: np.stack([d['pose2d'] for d in part_detections], axis=0) if len(part_detections) > 0 else np.empty((0, num_joints[part], 2), dtype=np.float32) for part, part_detections in detections.items()}
            scores = {part: [d['score'] for d in part_detections] for part, part_detections in detections.items()}
            output_image = visu.visualize_bodyhandface2d(np.asarray(image)[:, :, ::-1], det_poses2d, dict_scores=scores)
            return output_image, detections
        elif self.visualizer3D:
            print('Displaying results in 3D')
            viewer3d = visu3d.Viewer3d()
            img3d, img2d = viewer3d.plot3d(
                image,
                bodies={'pose3d': np.stack([d['pose3d'] for d in detections['body']]),
                        'pose2d': np.stack([d['pose2d'] for d in detections['body']])},
                hands={'pose3d': np.stack([d['pose3d'] for d in detections['hand']]),
                       'pose2d': np.stack([d['pose2d'] for d in detections['hand']])},
                faces={'pose3d': np.stack([d['pose3d'] for d in detections['face']]),
                       'pose2d': np.stack([d['pose2d'] for d in detections['face']])},
                body_with_wrists=body_with_wrists,
                body_with_head=body_with_head,
                interactive=False
            )
            return img2d, detections
        return detections


    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)

    def run_tracker(self):
        # video = cv2.VideoCapture(0)

        while True:
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

            # Perform pose landmark detection
            frame, detections = self.detect_pose(frame)
            print("Detections", detections)

            # Display frame
            if self.visualizer:
                cv2.imshow("Pose Detection", frame)

            # Wait for key press and retrieve ASCII code
            k = cv2.waitKey(1) & 0xFF

            # Check if key was 'ESC'
            if k == 27:
                break

            output_iu = self.create_iu(input_iu)
            output_iu.set_landmarks(image, detections, None)
            self.append(retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD))

    def setup(self):
        t = threading.Thread(target=self.run_tracker)
        t.start()