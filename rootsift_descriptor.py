import numpy as np


class RootSIFT:

    def __init__(self, descriptor, eps=1e-7):
        self.eps = eps
        self.descriptor = descriptor

    def detect(self, img, positions=None):
        assert positions is None
        return self.descriptor.detect(img, positions)

    def detectAndCompute(self, img, mask=None):
        kps, descs = self.descriptor.detectAndCompute(img, mask)
        if len(kps) == 0:
            return [], None

        descs /= (descs.sum(axis=1, keepdims=True) + self.eps)
        descs = np.sqrt(descs)
        return kps, descs

