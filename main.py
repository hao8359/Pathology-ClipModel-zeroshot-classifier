import torch
from PIL import Image
import open_clip
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # 1. Automatically detect GPU / CPU (prevents hardcoded .cuda() errors)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load model (Replace with the actual weight path on Bianca and remove author's cache_dir)
    model_path = "pathclip/pathclip-base.pt" 
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16', 
        pretrained=model_path,
        force_quick_gelu=True
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    model = model.to(device)
    model.eval()

    # 3. Load your 25,000 test set images (batch read using ImageFolder)
    dataset_path = "./lung_colon_image_set/Test Set" 
    dataset = datasets.ImageFolder(dataset_path, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    

    # 4. Create Text Prompts (matching your data classes)
    class_names = dataset.classes  # Automatically get folder class names
    #text_label_list = [f"An image of {c.replace('_', ' ')}" for c in class_names]
    #print(f"Class labels: {text_label_list}")
    # 1. 建立類別簡寫對應完整醫學專有名詞字典
    label_mapping = {
        "colon_aca": "colon adenocarcinoma",
        "colon_n": "normal colon tissue",
        "lung_aca": "lung adenocarcinoma",
        "lung_n": "normal lung tissue",
        "lung_scc": "lung squamous cell carcinoma"
    }

    # 2. 生成帶有病理語意的完整 Prompt
    text_label_list = [
        f"a histopathology image showing {label_mapping.get(c, c)}" 
        for c in class_names
    ]
    print(f"Class labels: {text_label_list}")

    text = tokenizer(text_label_list).to(device)

    # 5. Start Zero-shot evaluation (including feature extraction and matching)
    correct = 0
    total = 0

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Pre-extract text features
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Batch extract image features and compute similarity
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate Cosine similarity probabilities
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = torch.argmax(text_probs, dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # 6. Print final results
    acc = 100.0 * correct / total
    print(f"\n==========================================")
    print(f"Zero-Shot Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"==========================================")

if __name__ == "__main__":
    main()