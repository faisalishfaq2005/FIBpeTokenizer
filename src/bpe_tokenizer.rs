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
    final_vocabulary: HashSet<Rc<str>>,
    merge_rules: Vec<(Rc<str>,Rc<str>)>,
    initial_vocabulary_len:usize


}

impl BpeTokenizer {
    pub fn new(input_path:&str, vocab_size:usize, output_dir:&str) -> Self
    {
        BpeTokenizer{
            input_text_path:input_path.to_string(),
            target_vocab_size: vocab_size,
            output_dir:output_dir.to_string(),
            final_vocabulary:HashSet::new(),
            merge_rules:Vec::new(),
            initial_vocabulary_len:0

        }
    }
}


// impl BpeTokenizer {
//     pub fn read_file_and_return_words_as_vec_string(& self){
//         let file_data=fs::read_to_string(self.input_text_path.clone());
        

//         let raw_data:String =match file_data {
//             Ok(content) =>
//                 content,
            
//             Err(er) =>{
//                 eprintln!("Error reading file: {}", er);
//                 return;
//             }
//         };

//         raw_data
//     }
// }



impl BpeTokenizer {
    pub fn train(&mut self){
        let file_data=fs::read_to_string(self.input_text_path.clone());
        match file_data {
            Ok(content) =>
                {
                        let tokens:Vec<String>= content.split_whitespace().map(|s| s.to_string()).collect();
        
                        let mut vocab:HashMap<Vec<Rc<str>>,usize>=HashMap::new();
                        
                        let vocab_builder_start=Instant::now();

                        vocab_builder(&tokens, &mut vocab);
                        let vocab_builder_end=vocab_builder_start.elapsed();
                        println!("vocab builder took time: {:?}",vocab_builder_end);

                        let add_initial_vocab_start=Instant::now();
                        add_initial_unique_ch_in_final_vocab(&mut self.final_vocabulary, &vocab);
                        let add_initial_vocab_end=add_initial_vocab_start.elapsed();
                        println!("adding initial vocab took time: {:?}",add_initial_vocab_end);
                        
                        
                        self.initial_vocabulary_len=self.final_vocabulary.len();
                        
                        println!("{}",self.initial_vocabulary_len);

                        // let mut vocab_len=self.final_vocabulary.len();
                        // println!("{}" ,vocab_len);
                        // for ch in & final_vocabulary{
                        //     println!("{}",ch);
                        // }

                        // for (token, count) in &vocab {
                        //     println!("{:?}: {}", token, count);
                        // }
                        while  self.final_vocabulary.len()<self.target_vocab_size {
                        
                            
                            let max_pair: Option<(Rc<str>, Rc<str>)>= record_most_frequent_adjacent_pair(&vocab);
                            if max_pair.is_none(){
                                print!("no more pairs to merge");
                                break;
                            }

                            let max_pair_tuple:(Rc<str>,Rc<str>)=max_pair.unwrap();
                            let merged_pair=Rc::from(format!("{}{}", max_pair_tuple.0,max_pair_tuple.1));
                            if self.final_vocabulary.insert(Rc::clone(&merged_pair)){
                                self.merge_rules.push((Rc::clone(&max_pair_tuple.0),Rc::clone(&max_pair_tuple.1)));
                            }

                            replace_occurance_with_merged_pair(&mut vocab, &max_pair_tuple);
                            
                        }

                        // print!("final_vocabulary");
                        // for ch in & self.final_vocabulary{
                        //     println!("{}",ch);
                        // }
                        
                        // print!("now printing merged vocab");
                        // print!("");
                        // for (token, count) in &vocab {
                        //     println!("{:?}: {}", token, count);
                        // }

                        // print!("merge rules");
                        // for rule in merge_rules{
                        //     println!("{:?}",rule);
                        // }


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
        
        println!("{:?}",self.final_vocabulary);
        

        println!("{}",self.final_vocabulary.len());
    }
}


impl BpeTokenizer {
    pub fn initial_vocabulary_size(& self){
        println!("{}",self.initial_vocabulary_len);
    }
}

pub fn add_initial_unique_ch_in_final_vocab(final_vocabulary: &mut HashSet<Rc<str>>,vocab:& HashMap<Vec<Rc<str>>,usize>){
    for word in vocab.keys(){
        for token in word{
            final_vocabulary.insert(Rc::clone(token));
        }
    }
}






pub fn vocab_builder(pretokenized_text:&Vec<String>,vocab:&mut HashMap<Vec<Rc<str>>,usize>)   {
    
    for word in pretokenized_text{
        let mut char_seq:Vec<Rc<str>> = word.chars().map(|c| Rc::from(c.to_string())).collect();
        char_seq.push(Rc::from("</w>"));
        *vocab.entry(char_seq).or_insert(0) +=1;

    }


}

#[derive(Debug)]
enum TokenizerError{
    EmptyVocab,
}

pub fn record_most_frequent_adjacent_pair(vocab:&HashMap<Vec<Rc<str>>,usize>) -> Option<(Rc<str>,Rc<str>)>{
    let mut adjacent_pairs_frequency_hashmap: HashMap<(Rc<str>,Rc<str>),usize>=HashMap::new();
    
    for  (word,count) in vocab {
        if word.len() <2{
            continue;
        }
        for pair in word.windows(2)
        {
            let key = (Rc::clone(&pair[0]),Rc::clone(&pair[1]));
            *adjacent_pairs_frequency_hashmap.entry(key).or_insert(0) +=count;

        }
    }
    // for (tuple,count) in &adjacent_pairs_frequency_hashmap{
    //     println!("{:?}: {}", tuple, count);
    // }
    let max_pair=adjacent_pairs_frequency_hashmap.into_iter().max_by_key(|entry| entry.1).map(|(pair,_count)| pair);
    max_pair
}


pub fn replace_occurance_with_merged_pair(vocab:&mut HashMap<Vec<Rc<str>>,usize>,max_pair:& (Rc<str>,Rc<str>)){
    let mut updated_vocab:HashMap<Vec<Rc<str>>,usize>=HashMap::new();

    for (word,&count) in vocab.iter(){
        if word.len()<2{
            continue;
        }
        let mut new_word:Vec<Rc<str>>=Vec::new();
        let mut i=0;
        while i<word.len() {
            if i<word.len()-1 && word[i]==max_pair.0 && word[i+1]==max_pair.1{
                let merged= Rc::from(format!("{}{}", word[i], word[i + 1]));
                new_word.push(merged);
                i+=2;
            }
            else{
                new_word.push(Rc::clone(&word[i]));
                i+=1;
            }
                
        }
        *updated_vocab.entry(new_word).or_insert(0) += count;

    }
   vocab.clear();
   vocab.extend(updated_vocab);
}

