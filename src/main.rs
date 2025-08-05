use core::hash;
use std::{collections::{HashMap, HashSet}, fmt::format, usize};
use std::fs;
use std::io;
mod bpe_tokenizer;

use bpe_tokenizer::{BpeTokenizer};
use std::time::Instant;
mod tokenizer_tables;



fn main() {
    let mut tokenizer=BpeTokenizer::new("corpus.txt", 20000, "out_dir");
    let start=Instant::now();

    tokenizer.train();
    let duration=start.elapsed();
    println!("training time completed in : {:?}",duration);
    //tokenizer.display();

    

}

