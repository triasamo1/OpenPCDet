from functools import partial

import numpy as np
import numpy as np
import torch
from pcdet.models import load_data_to_gpu
import time
from icecream import ic


class CenterExtractor(object):
    def __init__(self, extractor_configs, training, run_once):
        self.training = training
        self.center_extractor_queue = []
        self.run_once = run_once
        for cur_cfg in extractor_configs:
            cur_extractor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.center_extractor_queue.append(cur_extractor)


    def extract_pointpillars_centers(self, data_dict=None, preds=None, config=None):
        if data_dict is None:
            ic('Extractor Check: ok')
            return partial(self.extract_pointpillars_centers, config=config)
        #ic(data_dict.keys())
        #if preds is not None:
            #ic(data_dict.keys())
            # dict_keys(['points', 'frame_id', 'gt_boxes', 'use_lead_xyz', 'voxels', 'voxel_coords', 'voxel_num_points', 'image_shape', 'batch_size', 'pillar_features', 'spatial_features', 'spatial_features_2d', 'batch_cls_preds', 'batch_box_preds', 'cls_preds_normalized'])
            #ic(data_dict['batch_box_preds'])
            #ic(preds)
            #for idx,each in pred.items():
            #    ic(data_dict.keys())    
        MODEL_PATH = '/home/triasamo/entire_model.pth'
        
        # load pretrained model to GPU and deactivate backprop
        
    
        return data_dict


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
                +
                Pointpillars predicted centers
        """

        for cur_extractor in self.center_extractor_queue:
            data_dict = cur_extractor(data_dict=data_dict)

        return data_dict
