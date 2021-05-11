from abc import ABC

import numpy as np
import torch
import os

from commons import preprocess_clip_metafunc, preprocess_frame_metafunc, sample_clips_metafunc

class FeatureExtractor(ABC):
    def __init__(self, clip_num, frame_num, stride, mean, std, resize_to, crop_to, *, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    @property
    def sample_func(self):
        if not hasattr(self, '_sample_func'):
            raise NotImplementedError("You should define 'sample_func'")
        return self._sample_func

    @sample_func.setter
    def sample_func(self, value):
        self._sample_func = value

    @property
    def preprocess_func(self):
        if not hasattr(self, '_preprocess_func'):
            raise NotImplementedError("You should define 'preprocess_func'")
        return self._preprocess_func

    @preprocess_func.setter
    def preprocess_func(self, value):
        self._preprocess_func = value

    def __call__(self, video_fpath, video_name, save_path):
        X, frame_number = self.sample_func(video_fpath)
        if X.shape[0] == 0:
            return None

        X = np.array([ self.preprocess_func(x) for x in X ])


        if len(X.shape) == 4: # ? -> ?
            X = X.transpose(( 0, 3, 1, 2 ))
        elif len(X.shape) == 5: # ? -> ?
            X = X.transpose(( 0, 4, 1, 2, 3 ))
        else:
            raise NotImplementedError("Unknown X of shape {}".format(X.shape))

        X = torch.from_numpy(X).float().cuda()

        feats = None

        for i in range(0, len(X), self.batch_size):
            X_batch = X[i: i + self.batch_size]
            feats_batch = self.model(X_batch)
            feats_batch = feats_batch.data.cpu().numpy()
            feats = feats_batch if feats is None else np.concatenate([ feats, feats_batch ], axis=0)
        save_full_path = os.path.join(save_path, video_name + '.npy')
        np.save(save_full_path, feats)

        return feats

class FeatureExtractor3D(FeatureExtractor):
    def __init__(self, clip_num, frame_num, stride, mean, std, resize_to, crop_to, **kwargs):
        super(FeatureExtractor3D, self).__init__(clip_num, frame_num, stride, mean, std, resize_to, crop_to, **kwargs)

        self.sample_func = sample_clips_metafunc(clip_num, frame_num, stride)
        self.preprocess_func = preprocess_clip_metafunc(mean, std, resize_to, crop_to)
