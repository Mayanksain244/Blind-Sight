import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from get_loader import get_loader

def load_model(checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    # First get the vocabulary from the dataset
    _, dataset = get_loader(
        root_folder="flickr/images",
        annotation_file="flickr/captions.txt",
        transform=None,
        num_workers=1,
    )

    # Model parameters (must match training parameters)
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2

    # Initialize model and load weights
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, dataset

def generate_caption(image_path, model, dataset, device):
    """
    Generate a caption for a given image
    """
    # Prepare the image transform (should match what the model was trained with)
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load and transform image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}"

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        caption = model.caption_image(image_tensor, dataset.vocab)

    # Filter out special tokens and join words
    caption = [word for word in caption if word not in ["<SOS>", "<EOS>", "<PAD>"]]
    caption = ' '.join(caption)

    return caption


def main():
    # Specify the paths directly in the script
    # image_path = "D:\\Minor2\\Work\\AutoImageCaption\\New\\flickr8k\\images\\3741462565_cc35966b7a.jpg"
    # image_path = "D:\\Minor2\\Work\\AutoImageCaption\\New\\flickr8k\\images\\3729405438_6e79077ab2.jpg"
    # image_path = "D:\\Minor2\\Work\\AutoImageCaption\\New\\Tests\\1.jpg"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-03-03 182422.png"
    image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-04-01 143338.png"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-03-31 140608.png"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Camera Roll\\WIN_20250227_12_50_39_Pro.jpg"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Camera Roll\\WIN_20250322_12_21_01_Pro.jpg"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Camera Roll\\WIN_20250322_12_17_45_Pro.jpg"
    # image_path = "C:\\Users\\mayan\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-03-06 133444.png"

    # model_checkpoint_path = "models\\3L_52_my_checkpoint.pth.tar"
    model_checkpoint_path = "my_checkpoint.pth.tar" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset = load_model(model_checkpoint_path, device)
    caption = generate_caption(image_path, model, dataset, device)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()