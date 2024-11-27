# This code generates Grad-CAM visualizations for a given image and model.

# 1. Load the Image and Preprocess:
#   - Load the image and resize it to the model's input size.
#   - Preprocess the image according to the model's requirements (e.g., normalization, 
#     channel order).

# 2. Forward Pass and Gradient Calculation:
#   - Forward pass the image through the model to obtain the output logits.
#   - Calculate the gradients of the target class's logit with respect to the activations
#     of the target layer (usually the last convolutional layer).

# 3. Compute Weighted Activations:
#   - Calculate the weighted sum of the target layer activations, using the gradients
#     as weights. This highlights the regions that most influenced the prediction.

# 4. Generate the Heatmap:
#   - Upsample the weighted activations to the original image size.
#   - Apply a ReLU activation to ensure non-negative values.
#   - Normalize the heatmap to a range of [0, 1].

# 5. Overlay Heatmap on Original Image:
#   - Overlay the heatmap on the original image to visualize the important regions.
#   - Adjust the alpha value to control the transparency of the heatmap.

# 6. Display or Save the Visualization:
#   - Display the visualization using a plotting library like Matplotlib.
#   - Save the visualization as an image file (e.g., PNG, JPEG).

# Note:
#   - The choice of the target layer can influence the granularity of the visualization.
#   - Experiment with different layers to obtain insights at different levels of abstraction.
#   - Consider using Grad-CAM++ for more precise visualizations, especially in cases where
#     the target class is not the most confident prediction.
from collections import OrderedDict
import numpy as np
import torch
import cv2
from torch.nn import functional as F

def normalize(input):
    output = (input - np.min(input)) / (np.max(input) - np.min(input))
    return output

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))

def save_gradcam(filename, gcam, raw_image):
    
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    #cv2.imshow(gcam)
    
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))
    cv2.rectangle(gcam,(384,0),(510,128),(0,255,0),3)
    plt.imshow(np.uint8(gcam))
    plt.show()

def save_raw_image(filename, raw_image):
    cv2.imwrite(filename, np.uint8(raw_image))


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

class GradCAM(_PropagationBase):
    def __init__(self, model):
        super(GradCAM, self).__init__(model)
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
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)
        gcam = (fmaps[0] * weights[0]).sum(dim=0)
        
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()
        return gcam.detach().cpu().numpy()
     