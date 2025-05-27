import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from torch.nn.functional import cosine_similarity


def get_image_embedding(image_path, transform, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).squeeze()
    return embedding


def compare_images(img1_path, img2_path, transform, model):
    emb1 = get_image_embedding(img1_path, transform, model)
    emb2 = get_image_embedding(img2_path, transform, model)
    similarity = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return similarity


def resnet(paired):
    filtered_pairs = []

    model = resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # (розмір, нормалізація під ImageNet)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for i, (left, right) in enumerate(paired):
        img1 = left["image"]
        img2 = right["image"]

        print(f"\nPair {i+1}:")

        similarity = compare_images(img1, img2, transform, model)
        print(f"Cosine Similarity: {similarity:.4f}")

        if similarity > 0.8:
            filtered_pairs.append((left, right))
            print("Ймовірно, це той самий об'єкт.")
        else:
            print("Ймовірно, це різні об'єкти.")

    return filtered_pairs
