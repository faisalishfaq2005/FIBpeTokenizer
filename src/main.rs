use core::hash;
use std::{collections::{HashMap, HashSet}, fmt::format, usize};
use std::fs;
use std::io;
mod bpe_tokenizer;

use bpe_tokenizer::{BpeTokenizer,Encoder};
use std::time::Instant;

//use crate::bpe_tokenizer::Encoder;
mod tokenizer_tables;



fn main() {
    let mut tokenizer=BpeTokenizer::new("corpus.txt", 4000, "out_dir");
    let start=Instant::now();

    tokenizer.train();
    let duration=start.elapsed();
    println!("training time completed in : {:?}",duration);

    //tokenizer.display();
    let encoding_text:&str="return steal something business free Henry";
    let encoded_text:Encoder=tokenizer.encode(&encoding_text);
    tokenizer.get_id_by_token("return</w>".to_string());
    tokenizer.get_id_by_token("something</w>".to_string());
    tokenizer.get_merge_pair_from_pairs_to_merge_and_ranks((3279,221));
    println!("{:?}",encoded_text.tokens);
    println!("{:?}",encoded_text.ids);
    println!("Encoded ")


//give something like bool option wheather user wants the trained tokenized to be saved, meaning the file thing.
    

    

}

