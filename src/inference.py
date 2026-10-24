import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.config_parser import load_config
from core.utils import SquarePad, load_model_weights
from core.model.swin_t import build_swin_t
from core.grad import SwinGradCAM

def main():
    config = load_config('config.yml')
    os.makedirs(config.inference.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_swin_t(len(config.inference.class_names))
    model = load_model_weights(model, config.inference.model_path, device)
    model = model.to(device)
    model.eval()

    inference_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((config.dataset.image_size, config.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(config.inference.image_path).convert('RGB')
    input_tensor = inference_transforms(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = config.inference.class_names[predicted_idx.item()]
    print(f"Prediction: {predicted_class} ({confidence.item() * 100:.2f}%)")

    # Grad-CAM extraction
    target_layer = model.features[-1]
    grad_cam = SwinGradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor, predicted_idx.item())

    # Visualization
    heatmap_resized = cv2.resize(heatmap, (config.dataset.image_size, config.dataset.image_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_tensor = input_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denorm = np.clip(std * img_tensor + mean, 0, 1)
    
    overlay = cv2.addWeighted(np.uint8(255 * img_denorm), 0.6, heatmap_colored, 0.4, 0)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_denorm)
    axs[1].imshow(heatmap_resized, cmap='jet')
    axs[2].imshow(overlay)
    
    base_name = os.path.basename(config.inference.image_path)
    save_path = os.path.join(config.inference.output_dir, f"cam_{base_name}")
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")

if __name__ == '__main__':
    main()