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
    let mut tokenizer=BpeTokenizer::new("corpus.txt", 20000,PreTokenization::Punctuation,special_tokens,SpecialTokenRemovalMethod::AhoCorasick,true,Some("out_dir")); //set mechanism for default valies for special tokens wich is empty list and for special token remover fn
    let start=Instant::now();


    tokenizer.train().unwrap();
    let duration=start.elapsed();
    println!("training time completed in : {:?}",duration);


    //from pretrained
    // let mut tokenizer=BpeTokenizer::new_from_pretrained("out_dir");
    // let r=tokenizer.train();
    // match r {
    //     Ok(())=> {},
    //     Err(error)=> eprintln!("error: {}",error)
        
    // }
    
    

    //tokenizer.display();
    let encoding_text:&str="<pad> we are going i'm to the supermarket <pad>, on Friday, and <mask> Wednesday. on the plane there, will be some of magic but <eow>" ;
    let encoded_text:Encoder=tokenizer.encode(&encoding_text).unwrap();
    //let get_token=tokenizer.get_id_by_token("flsjfs".to_string());
   
    println!("{:?}",encoded_text.tokens);
    println!("{:?}",encoded_text.ids);
    
    println!("Encoded ");
    let decoded=tokenizer.decode(&encoded_text.ids).unwrap();
    println!("decoded text: {}",decoded);
    println!("{:?}",encoded_text.token_types);
    let t_type=encoded_text.get_token_type("to</w>");
    println!("{:?}",t_type);
    


    

}






//make the match statements separate for each pretokenizer type in the decoder fn
// do the comment written after tokenizer::new() in main
//make api wrappers for python, will olso need to add error handling and other things in python style for that
//publish the library
//make the github readme
//make the documentation on lovable

