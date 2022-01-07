import torch.nn
import torchvision as tv
import numpy as np
from utils import Timer

from mit_semseg.models import ModelBuilder, SegmentationModule

from PIL import Image


net_encoder = ModelBuilder.build_encoder(
    arch='resnet18dilated',
    fc_dim=512,
    weights='.semseg/encoder_epoch_20.pth')


net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=512,
    num_class=150,
    weights='.semseg/decoder_epoch_20.pth',
    use_softmax=True)


def get_nonsky_mask(np_image, height, width, use_cuda=False):

    Timer.start_check_point("sky masking")

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    semseg_model = segmentation_module
    if use_cuda:
        semseg_model = semseg_model.cuda()
    pil_to_tensor = tv.transforms.Compose([
        tv.transforms.Resize((height, width)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    PIL_image = Image.fromarray(np.uint8(np_image)).convert('RGB')
    img_data = pil_to_tensor(PIL_image)
    if use_cuda:
        img_data = img_data.cuda()
    singleton_batch = {'img_data': img_data[None]}
    output_size = img_data.shape[1:]
    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = semseg_model(singleton_batch, segSize=output_size)
        # Get the predicted scores for each pixel
        _, pred = torch.max(scores, dim=1)
        pred = pred.detach().cpu()[0].numpy()
        nonsky_mask = pred != 2

    Timer.end_check_point("sky masking")

    return nonsky_mask
