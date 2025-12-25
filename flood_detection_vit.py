import os
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import evaluate
import numpy as np
import kagglehub

# 1. Configuration
MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k" # Base ViT model
BATCH_SIZE = 8 # Adjust based on memory
LEARNING_RATE = 5e-3
NUM_EPOCHS = 3 # Small number for demonstration

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Loading Functions
def load_data(data_dir):
    """
    Loads images and labels from the directory.
    Assumes structure: data/train, data/test
    Label Convention: Filenames ending with '_1.png' are flooded (1), others are non-flooded (0).
    """
    image_paths = []
    labels = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                
                # Determine label: 1 if flooded (_1.png), 0 otherwise
                if file.endswith("_1.png"):
                    labels.append(1)
                else:
                    labels.append(0)
    
    return image_paths, labels

def create_dataset(image_paths, labels):
    # Create a HF Dataset
    def gen():
        for path, label in zip(image_paths, labels):
            yield {"image": Image.open(path).convert("RGB"), "label": label}
            
    return Dataset.from_generator(gen)

from peft import LoraConfig, get_peft_model, PeftModel

# ... (Configuration constants remain) ...
SAVE_PATH = "./flood_detection_vit_lora"

# Check if model exists
if os.path.exists(SAVE_PATH):
    print(f"Found existing model at {SAVE_PATH}. Loading...")
    # Load Image Processor
    try:
        image_processor = ViTImageProcessor.from_pretrained(SAVE_PATH)
    except OSError:
        # Fallback if processor wasn't saved locally
        image_processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    # Load Base Model
    # We need the config to know the base model if passing just path, 
    # but simplest is to load base and then adapters
    base_model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2,
        id2label={0: "Non-Flooded", 1: "Flooded"},
        label2id={"Non-Flooded": 0, "Flooded": 1}
    )
    model = PeftModel.from_pretrained(base_model, SAVE_PATH)
    
    # Move to device
    model.to(device)
    print("Model loaded successfully.")

else:
    print("No existing model found. Starting training workflow...")

    # Download dataset
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("rahultp97/louisiana-flood-2016")
    print("Path to dataset files:", path)
    DATA_DIR = path

    print("Loading data files...")
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Data directories not found at {train_dir} or {test_dir}")
        print("Please make sure the 'data' folder contains 'train' and 'test' folders.")
    else:
        train_paths, train_labels = load_data(train_dir)
        test_paths, test_labels = load_data(test_dir)

        print(f"Found {len(train_paths)} training images and {len(test_paths)} test images.")

        print("Creating datasets...")
        train_dataset = create_dataset(train_paths, train_labels)
        test_dataset = create_dataset(test_paths, test_labels)

        # 3. Preprocessing
        print("Initializing Image Processor...")
        image_processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)

        def transform(example_batch):
            # Take a list of PIL images and turn them into pixel values
            inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
            inputs['label'] = example_batch['label']
            return inputs

        print("Applying transforms...")
        train_dataset = train_dataset.with_transform(transform)
        test_dataset = test_dataset.with_transform(transform)

        # 4. Model Setup with LoRA
        print("Setting up model...")
        # Load base model with binary classification head (num_labels=2)
        model = ViTForImageClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=2,
            id2label={0: "Non-Flooded", 1: "Flooded"},
            label2id={"Non-Flooded": 0, "Flooded": 1}
        )

        # Move model to device
        model.to(device)

        # Define LoRA Config
        print("Applying LoRA...")
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        
        # Wrap model
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        # 5. Training
        def compute_metrics(eval_pred):
            metric = evaluate.load("accuracy")
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        def collate_fn(batch):
            return {
                'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
                'labels': torch.tensor([x['label'] for x in batch])
            }

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            eval_strategy="epoch", 
            save_strategy="epoch",
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )

        print("Starting training...")
        trainer.train()

        print("Evaluating...")
        print(trainer.evaluate())

        # Save final model
        model.save_pretrained(SAVE_PATH)
        image_processor.save_pretrained(SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")



# 6. Inference Function (Gradio)
import gradio as gr

def predict_single_image(image, model, processor, device):
    """
    Predicts for a single image provided by Gradio.
    """
    if image is None:
        return "No image provided."
    
    # Preprocess
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = probs.max().item()
        predicted_class_idx = probs.argmax(-1).item()
        
    label = model.config.id2label[predicted_class_idx]
    return f"{label} ({score:.2%})"

if __name__ == "__main__":
    print("\nTraining complete. Launching Gradio interface...")
    
    # Prepare examples with absolute paths
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
        
    example_dir = os.path.join(base_dir, "examples")
    examples = []
    if os.path.exists(example_dir):
        for filename in ["flooded_1.jpg", "flooded_2.jpg", "not_flooded_1.jpg", "not_flooded_2.jpg"]:
            file_path = os.path.join(example_dir, filename)
            if os.path.exists(file_path):
                examples.append([file_path])
    
    # Create a wrapper function that has access to the global model/processor/device
    # This prevents NameError if the function is called when global scope is messy
    import functools
    predict_fn = functools.partial(predict_single_image, model=model, processor=image_processor, device=device)

    iface = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs="text",
        title="Flood Detection ViT",
        description="Upload an image to check if it shows flooding.",
        examples=examples if examples else None
    )
    
    iface.launch(share=True)
