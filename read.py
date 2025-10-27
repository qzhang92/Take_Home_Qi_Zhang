import os
import re
import numpy as np
from gensim.models import Word2Vec

EMBEDDING_DIM = 100 # The chosen vector dimension

def custom_tokenizer(file_content: str) -> list:
    tokens = []
    
    # 1. Process line by line and remove comments (adjust comment markers as needed)
    for line in file_content.split('\n'):
        # Remove inline comments
        if ';' in line:
            line = line[:line.index(';')]
        if '#' in line:
            line = line[:line.index('#')]
            
        # 2. Clean up whitespace and special characters, then split
        # Replace commas with spaces for unified handling
        line = line.replace(',', ' ').strip()
        
        # 3. Split by space and normalize to uppercase
        line_tokens = [token.upper() for token in line.split() if token]
        tokens.extend(line_tokens)
        
    return tokens

# Collection of tokens lists from all files (the Word2Vec training corpus)
corpus_tokens = []
file_names = []
file_contents = {}
for i in range(1, 3):
    file_name = f"{i}.txt"
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
            file_contents[file_name] = content
            tokens = custom_tokenizer(content)
            if tokens:
                corpus_tokens.append(tokens)
                file_names.append(file_name)

    except FileNotFoundError:
        print(f"Warning: File {file_name} not found.")

if not corpus_tokens:
    raise ValueError("Corpus is empty. Please check file paths and content.")
    
print(f"Tokenization complete. Collected {len(corpus_tokens)} documents.")

print("2. Training Word2Vec model...")

# Initialize and train the Word2Vec model
# vector_size: embedding dimension
# window: context window size
# min_count: ignores vocabulary words with a frequency less than this
model = Word2Vec(
    sentences=corpus_tokens, 
    vector_size=EMBEDDING_DIM, 
    window=5, 
    min_count=1, 
    workers=4 # Use multiple cores for speedup
)

# Lock the model to stop further training, improving memory efficiency
model.init_sims(replace=True) 

print(f"Word2Vec model training complete. Vocabulary size: {len(model.wv.key_to_index)}.")

def generate_document_vector(tokens: list, model: Word2Vec) -> np.ndarray:
    """
    Generates a document vector by averaging the Word2Vec vectors of all tokens in the file.
    """
    valid_vectors = []
    
    # Iterate through each token in the file
    for token in tokens:
        # Check if the token is in the Word2Vec model's vocabulary
        if token in model.wv:
            valid_vectors.append(model.wv[token])
            
    if valid_vectors:
        # Return the average of all valid vectors
        return np.mean(valid_vectors, axis=0)
    else:
        # If the file is empty or has no known vocabulary, return a zero vector
        return np.zeros(model.vector_size)

print("3. Generating the final document feature matrix X...")

# Initialize the feature matrix X
# Matrix shape is (number of files, vector dimension)
X = np.zeros((len(corpus_tokens), EMBEDDING_DIM))

# Iterate through all documents, generate vectors, and populate matrix X
for i, tokens in enumerate(corpus_tokens):
    doc_vector = generate_document_vector(tokens, model)
    X[i] = doc_vector

print(f"Feature matrix X generation complete. Shape: {X.shape}")

# X is now the input data for the subsequent DBSCAN clustering.
# Each row of the matrix represents the 100-dimensional feature vector for one file.
