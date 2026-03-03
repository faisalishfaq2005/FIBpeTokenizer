# Python Example Usage

from fibpetokenizer import (
    BpeTokenizer,
    PreTokenization,
    SpecialTokenRemovalMethod
)

def main():
    # Define special tokens
    special_tokens = ["<pad>", "<mask>", "<unk>", "<eow>"]

    # Create tokenizer
    print("Creating tokenizer...")
    tokenizer = BpeTokenizer(
        input_path="corpus.txt",
        target_vocab_size=10000,
        pretokenization_type=PreTokenization.punctuation(),
        special_tokens=special_tokens,
        special_token_removal_method=SpecialTokenRemovalMethod.aho_corasick(),
        save_model=True,
        output_dir="python_output_dir"
    )

    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train()
    print("Training complete!")

    # Encode text
    text = "<pad> Hello, world! This is a test. <mask> How are you? <eow>"
    print(f"\nOriginal text: {text}")
    
    encoder = tokenizer.encode(text)
    
    print(f"\nTokens: {encoder.tokens}")
    print(f"Token IDs: {encoder.ids}")
    print(f"Token Types: {encoder.token_types}")

    # Get token type for a specific token
    if encoder.tokens:
        first_token = encoder.tokens[0]
        token_type = encoder.get_token_type(first_token)
        print(f"\nToken '{first_token}' is of type: {token_type}")

    # Decode back to text
    decoded = tokenizer.decode(encoder.ids)
    print(f"\nDecoded text: {decoded}")

    # Get ID and token
    try:
        pad_id = tokenizer.get_id_by_token("<pad>")
        print(f"\n'<pad>' token ID: {pad_id}")
        
        token = tokenizer.get_token_by_id(pad_id)
        print(f"Token for ID {pad_id}: {token}")
    except Exception as e:
        print(f"Error: {e}")

    # Load a pretrained tokenizer
    print("\n\nLoading pretrained tokenizer...")
    loaded_tokenizer = BpeTokenizer.from_pretrained("python_output_dir")
    
    # Test with loaded tokenizer
    test_text = "Testing the loaded tokenizer!"
    test_encoder = loaded_tokenizer.encode(test_text)
    print(f"\nTest text: {test_text}")
    print(f"Test tokens: {test_encoder.tokens}")
    print(f"Test IDs: {test_encoder.ids}")

if __name__ == "__main__":
    main()
