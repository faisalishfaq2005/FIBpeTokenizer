
pub enum PreTokenization{
    Whitespace,
    Punctuation,
    
}

impl PreTokenization{
    pub fn pre_tokenize(&self,text:&str)-> Vec<String>{
        match self {
            PreTokenization::Whitespace => text.split_whitespace().map(|s| s.to_string()).collect(),
            PreTokenization::Punctuation => {
                let mut final_tokens:Vec<String>=Vec::new();
                let words:Vec<&str> = text.split_whitespace().collect();

                for word in words{
                    if word.contains(|c:char| c.is_ascii_punctuation()){

                        let mut current_part=String::new();
                        for char in word.chars(){
                            if char.is_ascii_punctuation(){
                                if !current_part.is_empty(){
                                    final_tokens.push(current_part);
                                    current_part=String::new();
                                    
                                }
                                final_tokens.push(char.to_string());
                            }
                            else{
                                current_part.push(char);
                            }
                        }
                        if !current_part.is_empty(){
                            final_tokens.push(current_part);
                        }
                       
                    }
                    else{
                        final_tokens.push(word.to_string());
                    }
                }
                final_tokens
            }
            
        }
    }
}