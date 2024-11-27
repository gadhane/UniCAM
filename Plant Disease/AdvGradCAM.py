from collections import OrderedDict
import numpy as np
import torch
import cv2
from torch.nn import functional as F


class _PropagationBase(object):
    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.image = None

    def _encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.preds.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.image = image.requires_grad_()
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)
        return self.prob, self.idx
    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

class AdGradCAM(_PropagationBase):
    def __init__(self, model):
        super(AdGradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.detach()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].detach()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm

    def _compute_grad_weights(self, grads):
        # Apply a localized pooling operation to the gradients
        # The kernel size, stride, and padding should be chosen based on the desired
        # level of spatial granularity and the dimensions of the feature maps
        kernel_size = 2  # Example value, should be adjusted based on specific needs
        stride = 2       # Example value, can be adjusted
        padding = 0      # Example value, can be adjusted

        # The pooling operation could be average pooling or max pooling
        pooled_grads = F.avg_pool2d(grads, kernel_size, stride=stride, padding=padding)

        return pooled_grads


    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        
        # Directly using the raw gradients as weights, preserving spatial information
        # weights = grads

        # Get the raw gradient weights without any pooling
        weights = self._compute_grad_weights(grads)

        # Upscale weights to match the spatial dimensions of the feature maps
        # Using bilinear interpolation (you can also try nearest-neighbor)
        weights = F.interpolate(weights, size=(fmaps.size(2), fmaps.size(3)), mode='bilinear', align_corners=False)

        # Element-wise multiplication and sum across the feature map dimension
        gcam = torch.sum(fmaps[0] * weights[0], dim=0)

        # gcam = torch.sum(fmaps[0] * weights[0], dim=0)

        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam.detach().cpu().numpy()
