"""Minimal Grad-CAM implementation for PyTorch models."""
import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        # input_tensor: 1 x C x H x W
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        score = out[0, class_idx]
        score.backward(retain_graph=True)
        activ = self.activations[0]  # C x H x W
        grads = self.gradients[0]    # C x H x W
        weights = grads.mean(dim=(1, 2))  # C
        cam = (weights[:, None, None] * activ).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
