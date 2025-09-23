from sentence_transformers import SentenceTransformer

def main():
    """
    Downloads and caches the sentence-transformer model.
    Run this script once to ensure the model is available locally.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Attempting to download and cache model: {model_name}")
    
    try:
        # This line will trigger the download and save the model to the cache.
        model = SentenceTransformer(model_name)
        print("\nModel downloaded and cached successfully!")
        print(f"Model files are located in: {model.get_sentence_embedding_dimension()} dimensions")
    except Exception as e:
        print(f"\nAn error occurred during download: {e}")
        print("Please check your internet connection and firewall settings.")

if __name__ == "__main__":
    main()
