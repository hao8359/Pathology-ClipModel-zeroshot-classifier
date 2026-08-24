import os
import torch
from PIL import Image
import open_clip
import wandb

# ==========================================
# Configuration & Paths (Edit Here)
# ==========================================
MODEL_NAME = "ViT-B-16"
PRETRAINED_BASE = "openai"
CHECKPOINT_PATH = "./checkpoints/epoch_10.pt"
IMAGE_PATH = "/path/to/test_slide.jpg"  # Path to your test histopathology image

WANDB_PROJECT = "PathCLIP-PathCap-Eval"
WANDB_RUN_NAME = "zero_shot_eval_slide"

# Candidate class labels for Zero-Shot classification
CANDIDATE_LABELS = [
    "normal tissue",
    "invasive ductal carcinoma",
    "lymph node metastasis",
    "chronic inflammation"
]

# ==========================================
# 1. Initialize WandB
# ==========================================
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "model": MODEL_NAME,
        "pretrained": PRETRAINED_BASE,
        "checkpoint": CHECKPOINT_PATH,
        "image_path": IMAGE_PATH,
    }
)

# ==========================================
# 2. Setup Device & Load Model
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create model structure
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED_BASE
)

# Load fine-tuned checkpoint
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()
print("Model loaded successfully!")

# ==========================================
# 3. Load and Preprocess Image
# ==========================================
raw_image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = preprocess(raw_image).unsqueeze(0).to(device)

# ==========================================
# 4. Prepare Prompts & Tokenize
# ==========================================
prompts = [f"a histopathology slide of {label}" for label in CANDIDATE_LABELS]

tokenizer = open_clip.get_tokenizer(MODEL_NAME)
text_tokens = tokenizer(prompts).to(device)

# ==========================================
# 5. Feature Extraction & Similarity Computation
# ==========================================
autocast_device = "cuda" if device == "cuda" else "cpu"
with torch.no_grad(), torch.amp.autocast(autocast_device):
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tokens)

    # L2 Normalization
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Cosine Similarity & Softmax Probabilities
    similarity = (100.0 * image_features @ text_features.T)
    probabilities = similarity.softmax(dim=-1)[0].cpu().numpy()

# ==========================================
# 6. Terminal Print & WandB Logging
# ==========================================
print("\n" + "="*50)
print(" === Zero-Shot Pathology Classification Results ===")
print("="*50)

wandb_table = wandb.Table(columns=["Class Label", "Probability (%)"])
prob_dict = {}

top_idx = probabilities.argmax()
top_label = CANDIDATE_LABELS[top_idx]
top_prob = probabilities[top_idx] * 100.0

for label, prob in zip(CANDIDATE_LABELS, probabilities):
    percentage = prob * 100.0
    print(f"Class: {label:<30} Probability: {percentage:.2f}%")
    wandb_table.add_data(label, round(percentage, 2))
    prob_dict[f"probabilities/{label}"] = percentage

print("="*50)
print(f"Top Prediction: {top_label} ({top_prob:.2f}%)\n")

# Log results to WandB dashboard
wandb.log({
    "eval/results_table": wandb_table,
    "eval/test_image": wandb.Image(
        raw_image, 
        caption=f"Predicted: {top_label} ({top_prob:.2f}%)"
    ),
    "eval/top_confidence": top_prob,
    **prob_dict
})

wandb.finish()
print("Evaluation complete")