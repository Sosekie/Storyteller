from typing import List, Dict, Any
from collections import Counter
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from skimage.metrics import structural_similarity as ssim
import spacy

# Load spaCy language model
nlp = spacy.load("en_core_web_sm")


def compute_llm_metrics(story: List[str]) -> Dict[str, Any]:
    """
    Compute metrics for the LLM-generated story.
    Metrics include perplexity (approximation), vocabulary diversity, and average sentence length.
    """
    # Flatten the story into a single string
    full_story = " ".join(story)

    # Tokenize the story using spaCy
    tokens = [token.text for token in nlp(full_story)]
    num_tokens = len(tokens)

    # Vocabulary diversity
    vocab = Counter(tokens)
    vocab_size = len(vocab)
    vocab_diversity = vocab_size / num_tokens if num_tokens > 0 else 0

    # Average sentence length
    avg_sentence_length = num_tokens / len(story) if len(story) > 0 else 0

    # Perplexity approximation (based on token frequencies)
    token_probs = [count / num_tokens for count in vocab.values()]
    perplexity = math.exp(-sum(p * math.log(p) for p in token_probs if p > 0))

    return {
        "vocab_diversity": vocab_diversity,
        "avg_sentence_length": avg_sentence_length,
        "perplexity": perplexity,
    }


def compute_story_context_fid(story: List[str]) -> Dict[str, Any]:
    """
    Compute context-FID and sentence similarity for the story.
    """
    # Load a pre-trained model for sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings for each sentence
    embeddings = model.encode(story)

    # Pairwise cosine similarity
    num_sentences = len(embeddings)
    similarities = []
    for i in range(num_sentences - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
        )
        similarities.append(sim)

    # FID-like metric: standard deviation of similarities
    context_fid = np.std(similarities)

    # Average similarity
    avg_similarity = np.mean(similarities)

    return {
        "context_fid": context_fid,
        "avg_similarity": avg_similarity,
    }


from typing import List, Dict, Any
from collections import Counter
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def compute_image_metrics(images: List[Any]) -> Dict[str, Any]:
    """
    Compute metrics for LDM-generated images.
    Metrics include FID (mocked with embeddings), SSIM between images, and image diversity.
    """
    # Convert PIL images to NumPy arrays
    images = [np.array(image) for image in images]

    # Compute embeddings for images (mocked with pixel flattening for simplicity)
    embeddings = [image.flatten() for image in images]

    # Pairwise SSIM between images
    num_images = len(images)
    ssim_scores = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            # Ensure window size is valid for the image
            min_size = min(images[i].shape[0], images[i].shape[1])
            win_size = min(7, min_size) if min_size >= 7 else min_size
            ssim_score, _ = ssim(
                images[i], images[j],
                full=True,
                channel_axis=-1,
                win_size=win_size
            )
            ssim_scores.append(ssim_score)

    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0

    # Diversity metric: standard deviation of embedding distances
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)
    diversity = np.std(distances) if distances else 0

    return {
        "avg_ssim": avg_ssim,
        "diversity": diversity,
    }
