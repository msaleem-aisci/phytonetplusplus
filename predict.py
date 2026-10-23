import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.utils import SquarePad, SwinGradCAM, load_model_weights
from src.model import build_swin_model

def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Grad-CAM for Swin-T")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    
    # Allow CLI overrides
    parser.add_argument('--image_path', type=str, default=None, help='Path to target image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    image_path = args.image_path or config['inference']['image_path']
    model_path = args.model_path or config['inference']['model_path']
    output_dir = args.output_dir or config['inference']['output_dir']
    class_names = config['inference']['class_names']
    image_size = config['dataset']['image_size']

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 1. Prepare Model
    model = build_swin_model(len(class_names))
    model = load_model_weights(model, model_path, device)
    model = model.to(device)
    model.eval()

    # 2. Prepare Image
    inference_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    input_tensor = inference_transforms(img).unsqueeze(0).to(device)

    # 3. Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = class_names[predicted_idx.item()]
    print(f"Predicted Class: {predicted_class} ({confidence.item() * 100:.2f}%)")

    # 4. Generate Grad-CAM Heatmap
    target_layer = model.features[-1]
    grad_cam = SwinGradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor, predicted_idx.item())

    # 5. Visualization Processing
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_tensor = input_tensor.squeeze().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_denorm = std * img_tensor + mean
    img_denorm = np.clip(img_denorm, 0, 1)
    img_denorm_uint8 = np.uint8(255 * img_denorm)

    overlay = cv2.addWeighted(img_denorm_uint8, 0.6, heatmap_colored, 0.4, 0)

    # 6. Plot and Save
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img_denorm)
    axs[0].set_title('Original Padded Image')
    axs[0].axis('off')

    axs[1].imshow(heatmap_resized, cmap='jet')
    axs[1].set_title('Grad-CAM Heatmap')
    axs[1].axis('off')

    axs[2].imshow(overlay)
    axs[2].set_title(f'Overlay (Prediction: {predicted_class})')
    axs[2].axis('off')

    plt.tight_layout()
    
    # Extract original filename for the output plot
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"cam_{base_name}")
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    
if __name__ == '__main__':
    main()