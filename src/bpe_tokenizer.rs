use core::hash;
use std::{collections::{HashMap, HashSet}, fmt::format, usize};
use std::fs;
use std::io;
use std::time::Instant;
use std::rc::Rc;
use crate::tokenizer_tables::TokenTable;



//olso add the pre_tokenization techinque in the struct so you can give it to the train function which will get the raw data and then it will split data according to the given technique
pub struct BpeTokenizer{
    input_text_path:String,
    target_vocab_size:usize,
    output_dir:String,
    token_table:TokenTable,
    merge_rules: Vec<(u32,u32)>,
    


}

impl BpeTokenizer {
    pub fn new(input_path:&str, vocab_size:usize, output_dir:&str) -> Self
    {
        BpeTokenizer{
            input_text_path:input_path.to_string(),
            target_vocab_size: vocab_size,
            output_dir:output_dir.to_string(),
            token_table:TokenTable::new(),
            merge_rules:Vec::new(),
            

        }
    }
}






impl BpeTokenizer {
    pub fn train(&mut self){
        let file_data=fs::read_to_string(self.input_text_path.clone());
        match file_data {
            Ok(content) =>
                {
                        let tokens:Vec<String>= content.split_whitespace().map(|s| s.to_string()).collect();
        
                        let mut vocab:HashMap<Vec<u32>,usize>=HashMap::new();
                        
                        let vocab_builder_start=Instant::now();

                        vocab_builder(&tokens, &mut vocab,&mut self.token_table);
                        let vocab_builder_end=vocab_builder_start.elapsed();
                        println!("vocab builder took time: {:?}",vocab_builder_end);

                    
                        
                

                        
                        while  self.token_table.get_len() < self.target_vocab_size {
                        
                            
                            let max_pair: Option<(u32, u32)>= record_most_frequent_adjacent_pair(&vocab);
                            if max_pair.is_none(){
                                print!("no more pairs to merge");
                                break;
                            }

                            let max_pair_tuple:(u32,u32)=max_pair.unwrap();
                            let token1 = self.token_table.get_token(max_pair_tuple.0).expect("Token1 not found");
                            let token2 = self.token_table.get_token(max_pair_tuple.1).expect("Token2 not found");

                            let merged_pair=format!("{}{}", token1,token2);
                            let merged_id=self.token_table.get_or_insert_id(&merged_pair);
                            self.merge_rules.push(max_pair_tuple);

                            replace_occurance_with_merged_pair(&mut vocab, &max_pair_tuple,merged_id);
                            
                        }

                       


                }
            
            Err(er) =>{
                eprintln!("Error reading file: {}", er);
                return;
            }
        };

    
       
        
    }
    
}

impl BpeTokenizer {
    pub fn display(& self){
        
        println!("{:?}",self.token_table.tokens());
        

        println!("{}",self.token_table.get_len());
    }
}











pub fn vocab_builder(pretokenized_text:&[String],vocab:&mut HashMap<Vec<u32>,usize>, table: &mut TokenTable )   {
    
    for word in pretokenized_text{
        let mut char_seq:Vec<u32> = word.chars().map(|c| {let ch=c.to_string(); table.get_or_insert_id(&ch)}).collect();
        let end_of_word_id=table.get_or_insert_id("</w>");
        char_seq.push(end_of_word_id);
        *vocab.entry(char_seq).or_insert(0) +=1;

    }


}

#[derive(Debug)]
enum TokenizerError{
    EmptyVocab,
}

pub fn record_most_frequent_adjacent_pair(vocab:&HashMap<Vec<u32>,usize>) -> Option<(u32,u32)>{
    let mut adjacent_pairs_frequency_hashmap: HashMap<(u32,u32),usize>=HashMap::new();
    
    for  (word,count) in vocab {
        if word.len() <2{
            continue;
        }
        for pair in word.windows(2)
        {
            let key = (pair[0],pair[1]);
            *adjacent_pairs_frequency_hashmap.entry(key).or_insert(0) +=count;

        }
    }
    // for (tuple,count) in &adjacent_pairs_frequency_hashmap{
    //     println!("{:?}: {}", tuple, count);
    // }
    let max_pair=adjacent_pairs_frequency_hashmap.into_iter().max_by_key(|entry| entry.1).map(|(pair,_count)| pair);
    max_pair
}


pub fn replace_occurance_with_merged_pair(vocab:&mut HashMap<Vec<u32>,usize>,max_pair:& (u32,u32),max_pair_id:u32){
    let mut updated_vocab:HashMap<Vec<u32>,usize>=HashMap::new();

    for (word,&count) in vocab.iter(){
        if word.len()<2{
            continue;
        }
        let mut new_word:Vec<u32>=Vec::new();
        let mut i=0;
        while i<word.len() {
            if i<word.len()-1 && word[i]==max_pair.0 && word[i+1]==max_pair.1{

                new_word.push(max_pair_id);
                i+=2;
            }
            else{
                new_word.push(word[i]);
                i+=1;
            }
                
        }
        *updated_vocab.entry(new_word).or_insert(0) += count;

    }
   vocab.clear();
   vocab.extend(updated_vocab);
}

