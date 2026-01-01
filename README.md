# 🌊 Flood Detection using Vision Transformers (ViT)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio&logoColor=white)
![LoRA](https://img.shields.io/badge/LoRA-PEFT-purple?style=for-the-badge)

A computer vision application that detects **flooded vs non-flooded scenes** in images using a fine-tuned **Vision Transformer (ViT)**.

The model is trained using **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning and includes an interactive **Gradio** interface for real-time inference.

---

## 🖼️ App Demo Screenshot

![Demo Interface](demo.png)

---

## ✨ Features

* **Vision Transformer Backbone:** Uses [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k).
* **Parameter-Efficient Training:** Fine-tunes ~0.6% of model parameters using LoRA.
* **Automated Dataset Handling:** Downloads the *Louisiana Flood 2016* dataset automatically via `kagglehub`.
* **Interactive Inference:** Upload images and get predictions via a Gradio UI.
* **Deployment Ready:** Compatible with Hugging Face Spaces.

---

## 🛠️ Tech Stack

* **Model:** Vision Transformer (ViT-Base)
* **Fine-Tuning:** LoRA (PEFT)
* **UI:** Gradio
* **Frameworks:** PyTorch, Transformers, PEFT
* **Dataset:** Louisiana Flood 2016 (Kaggle)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/arman1o1/flood-detection-ViT.git
cd flood-detection-ViT
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script:

```bash
python flood_detection_vit.py
```

### What happens next?

1. **Model Check**

   * Looks for trained LoRA adapters in `./flood_detection_vit_lora`

2. **Training (if needed)**

   * Downloads the dataset automatically
   * Fine-tunes the ViT model for 3 epochs
   * Saves adapters locally

3. **Inference**

   * Launches a Gradio web interface (local or public link)
   * Upload images to classify flood vs non-flood scenes

---

## ⚙️ Technical Details

* **Base Model:** [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)
* **Task:** Binary Image Classification
* **LoRA Configuration:**

  * Rank: 16
  * Alpha: 16
  * Target Modules: Query / Value
* **Execution:** GPU recommended, CPU supported
* **Caching:** Trained adapters reused on subsequent runs

---

## ⚠️ Notes & Limitations

* First run may take time due to dataset download and training
* GPU significantly speeds up training
* Intended for research and experimentation, not production deployment

---

## 📄 License

This project is licensed under the **MIT License**.
