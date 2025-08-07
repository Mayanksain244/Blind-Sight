import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from get_loader import get_loader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec
import string
import re
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_model(checkpoint_path, device):
    """
    Load the trained model from checkpoint
    """
    # First get the vocabulary from the dataset
    _, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=None,
        num_workers=1,
    )

    # Model parameters (must match training parameters)
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1

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
    caption = [word for word in caption if word not in ["< SOS >", "<EOS>", "<PAD>"]]
    caption = ' '.join(caption)

    return caption

def preprocess_text(text):
    """
    Preprocess text by lowercasing, removing punctuation and stopwords
    """
    # Download stopwords if not already downloaded
    try:
        stopwords_list = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stopwords_list = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords_list]
    
    return words

def calculate_bleu_score(generated_caption, reference_captions):
    """
    Calculate the BLEU score for a generated caption against multiple reference captions
    """
    # Tokenize the captions
    generated_tokens = generated_caption.split()
    reference_tokens = [caption.split() for caption in reference_captions]

    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=SmoothingFunction().method1, weights=(1, 0, 0, 0))
    return bleu_score

def calculate_jaccard_similarity(generated_caption, reference_captions):
    """
    Calculate the Jaccard similarity between generated caption and reference captions
    """
    # Preprocess the captions
    generated_words = set(preprocess_text(generated_caption))
    
    # Calculate Jaccard similarity with each reference caption and take the maximum
    max_jaccard = 0
    for reference in reference_captions:
        reference_words = set(preprocess_text(reference))
        
        # Calculate Jaccard similarity: intersection over union
        if len(generated_words) == 0 and len(reference_words) == 0:
            jaccard = 1.0  # Both are empty sets
        else:
            intersection = len(generated_words.intersection(reference_words))
            union = len(generated_words.union(reference_words))
            jaccard = intersection / union
            
        max_jaccard = max(max_jaccard, jaccard)
    
    return max_jaccard

def train_word2vec_model(all_captions):
    """
    Train a Word2Vec model on all captions
    """
    # Preprocess all captions
    processed_captions = [preprocess_text(caption) for caption in all_captions]
    
    # Train Word2Vec model
    model = Word2Vec(sentences=processed_captions, vector_size=100, window=5, min_count=1, workers=4)
    
    return model

def get_sentence_embedding(w2v_model, words):
    """
    Get sentence embedding by averaging word vectors
    """
    word_vectors = []
    for word in words:
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])
    
    if len(word_vectors) == 0:
        return np.zeros(w2v_model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def calculate_wmd(w2v_model, generated_caption, reference_captions):
    """
    Calculate Word Mover's Distance between generated caption and reference captions
    """
    # Preprocess the captions
    generated_words = preprocess_text(generated_caption)
    
    if len(generated_words) == 0:
        return 0
    
    # Calculate WMD with each reference caption and take the minimum (best match)
    min_distance = float('inf')
    
    for reference in reference_captions:
        reference_words = preprocess_text(reference)
        
        if len(reference_words) == 0:
            continue
        
        # Check if all words are in the vocabulary
        if all(word in w2v_model.wv for word in generated_words) and all(word in w2v_model.wv for word in reference_words):
            try:
                distance = w2v_model.wv.wmdistance(generated_words, reference_words)
                if not np.isinf(distance):
                    min_distance = min(min_distance, distance)
            except Exception:
                # Skip if there's an error calculating WMD
                continue
    
    # Convert to similarity (1 - normalized distance)
    if min_distance == float('inf'):
        return 0
    
    # Normalize and convert to similarity (higher is better)
    similarity = np.exp(-min_distance)
    return similarity

def calculate_cosine_similarity(w2v_model, generated_caption, reference_captions):
    """
    Calculate cosine similarity between generated caption and reference captions
    """
    # Get embedding for generated caption
    generated_words = preprocess_text(generated_caption)
    generated_embedding = get_sentence_embedding(w2v_model, generated_words)
    
    # Calculate cosine similarity with each reference caption and take the maximum
    max_cosine = 0
    for reference in reference_captions:
        reference_words = preprocess_text(reference)
        reference_embedding = get_sentence_embedding(w2v_model, reference_words)
        
        # Ensure embeddings are not zero vectors
        if not np.all(generated_embedding == 0) and not np.all(reference_embedding == 0):
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(generated_embedding, reference_embedding)
            max_cosine = max(max_cosine, similarity)
    
    return max_cosine

def evaluate_model(model, dataset, device, image_folder, caption_file, sample_percentage=0.1):
    """
    Evaluate the model by generating captions for images and comparing them with reference captions
    """
    # Load the captions from the CSV file
    captions_df = pd.read_csv(caption_file)

    # Sample a percentage of the data
    sampled_df = captions_df.sample(frac=sample_percentage, random_state=42)

    # Prepare metrics
    metrics = {
        'bleu': 0,
        'jaccard': 0,
        'cosine': 0
    }
    num_images = 0

    # Group the captions by image ID
    grouped_captions = sampled_df.groupby('image')['caption'].apply(list).reset_index()
    
    # Collect all captions for training Word2Vec model
    all_captions = captions_df['caption'].tolist()
    print("Training Word2Vec model on all captions...")
    w2v_model = train_word2vec_model(all_captions)
    print("Word2Vec model trained.")

    # Use tqdm to show progress
    for index, row in tqdm(grouped_captions.iterrows(), total=len(grouped_captions), desc="Evaluating"):
        image_id = row['image']
        reference_captions = row['caption']
        image_path = os.path.join(image_folder, image_id)
        
        # Generate caption for the image
        generated_caption = generate_caption(image_path, model, dataset, device)
        
        if "Error loading image" in generated_caption:
            print(f"Skipping {image_id}: {generated_caption}")
            continue
            
        # Calculate metrics
        metrics['bleu'] += calculate_bleu_score(generated_caption, reference_captions)
        metrics['jaccard'] += calculate_jaccard_similarity(generated_caption, reference_captions)
        metrics['cosine'] += calculate_cosine_similarity(w2v_model, generated_caption, reference_captions)
        
        num_images += 1

    # Calculate averages
    for metric in metrics:
        metrics[metric] /= num_images

    return metrics

def main():
    # Specify the paths directly in the script
    # model_checkpoint_path = "models/2L_50_my_checkpoint_30k.pth.tar"
    # model_checkpoint_path = "models/2L_ALL_30_my_checkpoint.pth.tar"
    # model_checkpoint_path = "models/2L_40_my_checkpoint_30k.pth.tar"
    # model_checkpoint_path = "models/2L_30_my_checkpoint_30k.pth.tar"
    # model_checkpoint_path = "models/3L_20_my_checkpoint.pth.tar"
    # model_checkpoint_path = "models/32_my_checkpoint.pth.tar"
    model_checkpoint_path = "models/22_my_checkpoint.pth.tar"
    image_folder = "flickr8k/images"
    caption_file = "flickr8k/captions.txt"

    # Download NLTK resources if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model, dataset = load_model(model_checkpoint_path, device)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, dataset, device, image_folder, caption_file, sample_percentage=0.1)
    
    print("\nEvaluation Results:")
    print(f"BLEU-1 Score: {metrics['bleu']:.4f}")
    print(f"Jaccard Similarity: {metrics['jaccard']:.4f}")
    print(f"Cosine Similarity: {metrics['cosine']:.4f}")

if __name__ == "__main__":
    main()


# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# from model import CNNtoRNN
# from get_loader import get_loader
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import pandas as pd
# import os
# from tqdm import tqdm
# import random

# def load_model(checkpoint_path, device):
#     """
#     Load the trained model from checkpoint
#     """
#     # First get the vocabulary from the dataset
#     _, dataset = get_loader(
#         root_folder="flickr30k/images",
#         annotation_file="flickr30k/captions.txt",
#         transform=None,
#         num_workers=1,
#     )

#     # Model parameters (must match training parameters)
#     embed_size = 256
#     hidden_size = 256
#     vocab_size = len(dataset.vocab)
#     num_layers = 2

#     # Initialize model and load weights
#     model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["state_dict"])
#     model.eval()

#     return model, dataset

# def generate_caption(image_path, model, dataset, device):
#     """
#     Generate a caption for a given image
#     """
#     # Prepare the image transform (should match what the model was trained with)
#     transform = transforms.Compose(
#         [
#             transforms.Resize((299, 299)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )

#     # Load and transform image
#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         return f"Error loading image: {e}"

#     image_tensor = transform(image).unsqueeze(0).to(device)

#     # Generate caption
#     with torch.no_grad():
#         caption = model.caption_image(image_tensor, dataset.vocab)

#     # Filter out special tokens and join words
#     caption = [word for word in caption if word not in ["<SOS>", "<EOS>", "<PAD>"]]
#     caption = ' '.join(caption)

#     return caption

# def calculate_bleu_score(generated_caption, reference_captions):
#     """
#     Calculate the BLEU score for a generated caption against multiple reference captions
#     """
#     # Tokenize the captions
#     generated_tokens = generated_caption.split()
#     reference_tokens = [caption.split() for caption in reference_captions]

#     # Calculate BLEU score
#     bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=SmoothingFunction().method1 , weights=(1, 0, 0, 0))
#     return bleu_score

# def evaluate_model(model, dataset, device, image_folder, caption_file, sample_percentage=0.1):
#     """
#     Evaluate the model by generating captions for images and comparing them with reference captions
#     """
#     # Load the captions from the CSV file
#     captions_df = pd.read_csv(caption_file)

#     # Sample 10% of the data
#     sampled_df = captions_df.sample(frac=sample_percentage, random_state=42)

#     total_bleu_score = 0
#     num_images = 0

#     # Group the captions by image ID
#     grouped_captions = sampled_df.groupby('image')['caption'].apply(list).reset_index()

#     # Use tqdm to show progress
#     for index, row in tqdm(grouped_captions.iterrows(), total=len(grouped_captions), desc="Evaluating"):
#         image_id = row['image']
#         reference_captions = row['caption']
#         image_path = os.path.join(image_folder, image_id)
#         generated_caption = generate_caption(image_path, model, dataset, device)
#         bleu_score = calculate_bleu_score(generated_caption, reference_captions)
#         total_bleu_score += bleu_score
#         num_images += 1

#     average_bleu_score = total_bleu_score / num_images
#     return average_bleu_score

# def main():
#     # Specify the paths directly in the script
#     model_checkpoint_path = "models/2L_30_my_checkpoint_30k.pth.tar"
#     image_folder = "flickr30k/images"
#     caption_file = "flickr30k/captions.txt"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, dataset = load_model(model_checkpoint_path, device)
#     average_bleu_score = evaluate_model(model, dataset, device, image_folder, caption_file , sample_percentage=0.1)
#     print(f"Average BLEU Score: {average_bleu_score}")

# if __name__ == "__main__":
#     main()
