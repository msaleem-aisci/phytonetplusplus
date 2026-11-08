import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        pad_left = int((max_wh - w) / 2)
        pad_top = int((max_wh - h) / 2)
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode='constant')

def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state['model_state_dict'], best_model_path)

def load_model_weights(model, checkpoint_path, device):
    """Safely loads weights, stripping 'module.' prefix if saved via DataParallel."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    # If the state_dict is nested under 'model_state_dict' (from our Trainer class)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
        
    model.load_state_dict(clean_state_dict)
    return model

class SwinGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # Swin outputs (Batch, Height, Width, Channels). 
        # Permute to (Batch, Channels, Height, Width) for CNN-style math
        self.activations = output.permute(0, 3, 1, 2)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].permute(0, 3, 1, 2)

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().clone()
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap.cpu().numpy()