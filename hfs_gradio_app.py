import os
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from peft import PeftModel
import gradio as gr
import functools

# 1. Configuration
# The base model used during training (needed to initialize the architecture)
BASE_MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k" 
# The Hugging Face Hub repository ID where the adapter is hosted
ADAPTER_REPO_ID = "arman1o1/flood-detection-vit-lora"

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_inference_model(adapter_repo, base_checkpoint):
    """
    Loads the Base ViT model and merges it with the LoRA adapter from Hugging Face Hub.
    """
    print(f"Loading adapter from {adapter_repo}...")

    # 1. Load Image Processor
    try:
        # Try loading specific processor config if it was saved to the repo
        image_processor = ViTImageProcessor.from_pretrained(adapter_repo)
    except OSError:
        # Fallback to base model processor if not found in the adapter repo
        print("Processor not found in adapter repo, falling back to base configuration.")
        image_processor = ViTImageProcessor.from_pretrained(base_checkpoint)

    # 2. Load Base Model
    # Must specify label mappings exactly as they were during training
    print("Loading base model...")
    base_model = ViTForImageClassification.from_pretrained(
        base_checkpoint,
        num_labels=2,
        id2label={0: "Non-Flooded", 1: "Flooded"},
        label2id={"Non-Flooded": 0, "Flooded": 1}
    )

    # 3. Load and Activate LoRA Adapter
    print("Fetching and merging LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_repo)
    
    # Move to GPU/CPU and set to evaluation mode
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model, image_processor

# 2. Inference Function
def predict_single_image(image, model, processor, device):
    """
    Predicts class for a single PIL image.
    """
    if image is None:
        return "No image provided."
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply Softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top prediction
        score = probs.max().item()
        predicted_class_idx = probs.argmax(-1).item()
        
    # Map index to label
    label = model.config.id2label[predicted_class_idx]
    
    return f"{label} ({score:.2%})"

# 3. Main Execution
if __name__ == "__main__":
    # Load Model and Processor
    try:
        model, image_processor = load_inference_model(ADAPTER_REPO_ID, BASE_MODEL_CHECKPOINT)
        
        # Create a partial function to pass model/processor to Gradio
        predict_fn = functools.partial(
            predict_single_image, 
            model=model, 
            processor=image_processor, 
            device=device
        )

        # Optional: Check for example images to display in the UI
        examples = []
        example_dir = os.path.join(os.getcwd(), "examples")
        if os.path.exists(example_dir):
            for filename in os.listdir(example_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    examples.append([os.path.join(example_dir, filename)])

        print("Launching Gradio interface...")
        
        iface = gr.Interface(
            fn=predict_fn,
            inputs=gr.Image(type="pil", label="Upload Image"),
            outputs=gr.Text(label="Prediction"),
            title="Flood Detection ViT (Inference)",
            description="Upload an image to check if it shows flooding (Flooded) or not (Non-Flooded).",
            examples=examples if examples else None
        )
        
        iface.launch(server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        print(f"An error occurred: {e}")
