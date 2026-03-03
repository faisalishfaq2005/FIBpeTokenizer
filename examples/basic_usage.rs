use fibpetokenizer::{BpeTokenizer, PreTokenization, SpecialTokenRemovalMethod};

fn main() {
    // Define special tokens
    let special_tokens = vec![
        "<pad>".to_string(),
        "<mask>".to_string(),
        "<unk>".to_string(),
        "<eow>".to_string(),
    ];

    // Create tokenizer
    println!("Creating tokenizer...");
    let mut tokenizer = BpeTokenizer::new(
        "corpus.txt",
        10000,
        PreTokenization::Punctuation,
        special_tokens,
        SpecialTokenRemovalMethod::AhoCorasick,
        true,
        Some("example_output_dir"),
    );

    // Train the tokenizer
    println!("Training tokenizer...");
    tokenizer.train().expect("Training failed");
    println!("Training complete!");

    // Encode text
    let text = "<pad> Hello, world! This is a test. <mask> How are you? <eow>";
    println!("\nOriginal text: {}", text);

    let encoder = tokenizer.encode(text).expect("Encoding failed");

    println!("\nTokens: {:?}", encoder.tokens);
    println!("Token IDs: {:?}", encoder.ids);
    println!("Token Types: {:?}", encoder.token_types);

    // Get token type for a specific token
    if let Some(first_token) = encoder.tokens.first() {
        match encoder.get_token_type(first_token) {
            Ok(token_type) => println!("\nToken '{}' is of type: {:?}", first_token, token_type),
            Err(e) => println!("Error getting token type: {}", e),
        }
    }

    // Decode back to text
    let decoded = tokenizer.decode(&encoder.ids).expect("Decoding failed");
    println!("\nDecoded text: {}", decoded);

    // Get ID and token
    match tokenizer.get_id_by_token("<pad>".to_string()) {
        Ok(pad_id) => {
            println!("\n'<pad>' token ID: {}", pad_id);
            match tokenizer.get_token_by_id(pad_id) {
                Ok(token) => println!("Token for ID {}: {}", pad_id, token),
                Err(e) => println!("Error: {}", e),
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    // Load a pretrained tokenizer
    println!("\n\nLoading pretrained tokenizer...");
    let loaded_tokenizer = BpeTokenizer::new_from_pretrained("example_output_dir");

    // Test with loaded tokenizer
    let test_text = "Testing the loaded tokenizer!";
    let test_encoder = loaded_tokenizer
        .encode(test_text)
        .expect("Encoding with loaded tokenizer failed");
    println!("\nTest text: {}", test_text);
    println!("Test tokens: {:?}", test_encoder.tokens);
    println!("Test IDs: {:?}", test_encoder.ids);
}
