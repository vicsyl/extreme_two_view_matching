import torch
import torch.nn.functional as F


def upsample_bilinear(depth_data, height, width):
    upsampling = torch.nn.Upsample(size=(height, width), mode='bilinear')
    depth_data = upsampling(depth_data)
    return depth_data


def resample_nearest_numpy(data, height, width):

    if data.shape[0] == height:
        return data
    else:
        data_height = data.shape[0]
        data = torch.from_numpy(data)
        data = data.view(1, 1, data.shape[0], data.shape[1])
        if data_height < height:
            upsampling = torch.nn.Upsample(size=(height, width), mode='nearest')
            data = upsampling(data)
        else:
            data = F.interpolate(data, size=(height, width), scale_factor=None, mode='nearest', align_corners=None)
    ret = data.squeeze(dim=0).squeeze(dim=0).numpy()
    return ret
