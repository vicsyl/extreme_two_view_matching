import kornia as K
import kornia.feature as KF
from kornia.utils import batched_forward
from kornia_moons.feature import *
import numpy as np
import torch
from utils import Timer

"""
DISCLAIMER: taken from https://github.com/kornia/kornia-examples/blob/master/MKD_TFeat_descriptors_in_kornia.ipynb
"""


class HardNetDescriptor:

    def __init__(self, sift_descriptor, device: torch.device=torch.device('cpu')):
        self.sift_descriptor = sift_descriptor
        self.hardnet = KF.HardNet(True)
        self.device = device
        self.affine = KF.LAFAffNetShapeEstimator(True)
        self.orienter = KF.LAFOrienter(32, angle_detector=KF.OriNet(True))
        self.set_device_eval_to_nets([self.hardnet, self.affine, self.orienter], self.device)

    @staticmethod
    def set_device_eval_to_nets(nets: list, device):
        for net in nets:
            net.eval()
            if device == torch.device('cuda'):
                net.cuda()
            else:
                net.cpu()

    def detectAndCompute(self, img, mask=None, give_laffs=False, filter=None):
        Timer.start_check_point("HadrNet.detectAndCompute")
        assert mask is None
        kps = self.sift_descriptor.detect(img, None)
        if filter is not None:
            kps = kps[::filter]
        ret = self.get_local_descriptors(img, kps, compute_laffs=give_laffs)
        if len(ret) != 2:
            # corner case
            descs = np.zeros(0)
            laffs = np.zeros(0)
        else:
            descs, laffs = ret

        Timer.end_check_point("HadrNet.detectAndCompute")
        if give_laffs:
            return kps, descs, laffs
        else:
            return kps, descs

    def get_local_descriptors(self, img, cv2_sift_kpts, compute_laffs=False):
        if len(cv2_sift_kpts) == 0:
            return np.array([])

        # We will not train anything, so let's save time and memory by no_grad()
        with torch.no_grad():
            if len(img.shape) == 3:
                pass
            elif len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)
                img = np.repeat(img, 3, axis=2)
            else:
                raise Exception("Unexpected shape of the img: {}".format(img.shape))
            timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float() / 255.).to(self.device)
            lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts, device=self.device)

            if compute_laffs:
                lafs2 = self.affine(lafs, timg)
                lafs_to_use = self.orienter(lafs2, timg)
            else:
                lafs_to_use = lafs

            patches = KF.extract_patches_from_pyramid(timg, lafs_to_use, 32)

            B, N, CH, H, W = patches.size()
            patches = patches.view(B * N, CH, H, W)

            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :)
            descs = batched_forward(self.hardnet, patches, self.device, 128).view(B * N, -1)

        return descs.detach().cpu().numpy(), lafs_to_use.detach().cpu()
