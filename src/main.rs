use core::hash;
use std::{any::type_name, collections::{HashMap, HashSet}, fmt::format, usize};
use std::fs;
use std::io;
mod bpe_tokenizer;

use bpe_tokenizer::{BpeTokenizer,Encoder,SpecialTokenRemovalMethod};
use std::time::Instant;
mod pretokenizer;
use pretokenizer::PreTokenization;



//use crate::bpe_tokenizer::Encoder;
mod tokenizer_tables;



fn main() {
    let special_tokens:Vec<String>=vec!["<pad>".to_string(),"<mask>".to_string(),"<unk>".to_string(),"<eow>".to_string()];
    let special_tokens2:Vec<String>=Vec::new();
    let mut tokenizer=BpeTokenizer::new("corpus.txt", 4000,PreTokenization::Punctuation,special_tokens,SpecialTokenRemovalMethod::AhoCorasick,"out_dir"); //set mechanism for default valies for special tokens wich is empty list and for special token remover fn
    let start=Instant::now();

    tokenizer.train();
    let duration=start.elapsed();
    println!("training time completed in : {:?}",duration);

    //tokenizer.display();
    let encoding_text:&str="<pad> we are going i'm to the supermarket <pad>, on Friday, and <mask> Wednesday. on the plane there, will be some of magic but <eow>" ;
    let encoded_text:Encoder=tokenizer.encode2(&encoding_text);
   
    println!("{:?}",encoded_text.tokens);
    println!("{:?}",encoded_text.ids);
    
    println!("Encoded ");
    let decoded=tokenizer.decode(&encoded_text.ids);
    println!("decoded text: {}",decoded);
    println!("{:?}",encoded_text.token_types);
    let t_type=encoded_text.get_token_type("to</w>");
    println!("{:?}",t_type);


    
    


//give something like bool option wheather user wants the trained tokenized to be saved, meaning the file thing.
    
    



    

}


//convert to library
// handle file reading separately
//handle the errors and communicate them effectively
//handle match statements in decode fn for pre tokenization
// include the punctuation type in tokentype and add token.isasciipunctuation to check for speciltokens currently it is for only </w>

//make the match statements separate for each pretokenizer type in the decoder fn
// do the comment written after tokenizer::new() in main