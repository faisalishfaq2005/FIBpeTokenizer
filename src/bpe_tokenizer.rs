use core::hash;
use std::{collections::{BinaryHeap, HashMap, HashSet}, fmt::format, u32, usize};

use std::fs;
use std::io;
use std::time::Instant;
use std::rc::Rc;
use crate::tokenizer_tables::TokenTable;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Word {
    pub tokens: Vec<u32>,
    pub count: usize,
}

//olso make a function for getting id given a special token
//olso add the pre_tokenization techinque in the struct so you can give it to the train function which will get the raw data and then it will split data according to the given technique
pub struct BpeTokenizer{
    input_text_path:String,
    target_vocab_size:usize,
    output_dir:String,
    token_table:TokenTable,
    merge_rules: Vec<(u32,u32)>,
    ranks: HashMap<(u32,u32),usize>,
    pairs_to_merge: HashMap<(u32,u32),u32>
    
}

pub struct Encoder{
    pub original_text:String,
    pub tokens:Vec<String>,
    pub ids:Vec<u32>
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
            ranks:HashMap::new(),
            pairs_to_merge: HashMap::new()
            

        }
    }
}

impl Encoder{
    pub fn new() -> Self{
        Encoder{
        original_text:String::new(),
        tokens:Vec::new(),
        ids:Vec::new(),
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
        
                        let mut vocab:Vec<Word>=Vec::new();

                        let mut pair_occurances:HashMap<(u32,u32),HashSet<(usize,usize)>>=HashMap::new();

                        
                        let vocab_builder_start=Instant::now();

                        vocab_builder2(&tokens, &mut vocab,&mut self.token_table,&mut pair_occurances);
                        let vocab_builder_end=vocab_builder_start.elapsed();
                        println!("vocab builder took time: {:?}",vocab_builder_end);
                        

                        let build_initial_pair_start=Instant::now();
                        let  (mut pair_freq , mut heap)=build_initial_pair_stats(&vocab);
                        let build_initial_pair_end=build_initial_pair_start.elapsed();
                        println!("build initial pair took time: {:?}",build_initial_pair_end);

                    
                        
                

                        
                        while  self.token_table.get_len() < self.target_vocab_size {
                        
                            let max_pair=loop{
                                if let Some((freq,pair))=heap.pop(){
                                    if let Some(&actual_freq)=pair_freq.get(&pair){
                                        if freq==actual_freq{
                                            break Some(pair);
                                        }
                                    }
                                }
                                else {
                                    break None;
                                }
                            };
                           
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
                            update_pair_stats_after_merge(&vocab, &mut pair_freq, &mut heap, &max_pair_tuple, merged_id,& pair_occurances);
                           
                            replace_occurance_with_merged_pair(&mut vocab, &max_pair_tuple,merged_id,&mut pair_occurances);
                            
                        }

                        let (ranks,pairs_to_merge)=build_ranks_and_pairs_to_merge(&self.merge_rules, &self.token_table);
                        self.ranks=ranks;
                        self.pairs_to_merge=pairs_to_merge;

                       
                        println!("Checking if 'something</w>' exists during training...");
                        // Assuming get_id returns u32 (not Option<u32>)
                        let something_id = self.token_table.get_id("something</w>");
                        println!("something</w> ID during training: {}", something_id);

                        println!("But was it added through BPE? Checking merge rules...");
                        let mut found = false;
                        for &(a, b) in &self.merge_rules {
                            let token_a = self.token_table.get_token(a).expect("not found");
                            let token_b = self.token_table.get_token(b).expect("not found");
                            if format!("{}{}", token_a, token_b) == "something</w>" {
                                println!("Found merge rule: '{}' + '{}'", token_a, token_b);
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            println!("'something</w>' exists but no merge rule found! This is the bug.");
                            
                            // Let's also check what tokens 3279 and 221 actually are
                            let token_3279 = self.token_table.get_token(3279).expect("not found");
                            let token_221 = self.token_table.get_token(221).expect("not found");
                            println!("Token 3279: '{}'", token_3279);
                            println!("Token 221: '{}'", token_221);
                            println!("Their combination: '{}{}'", token_3279, token_221);
                        }
                }
            
            Err(er) =>{
                eprintln!("Error reading file: {}", er);
                return;
            }
        };

    
       
        
    }
    pub fn encode(&self, raw_encoding_text: &str) -> Encoder {
    let encoding_text_pretokenized: Vec<String> = raw_encoding_text.split_whitespace()
        .map(|s| s.to_string())
        .collect();
    let end_of_word_id: u32 = self.token_table.get_id("</w>");
    let mut encoder_object = Encoder::new();
    encoder_object.original_text = raw_encoding_text.to_string();
    let mut words: Vec<Vec<u32>> = Vec::new();


    for word in encoding_text_pretokenized.iter() {
        let mut char_ids: Vec<u32> = word.chars()
            .map(|c| {
                let ch = c.to_string();
                self.token_table.get_id(&ch)
            })
            .collect();
        char_ids.push(end_of_word_id);
        words.push(char_ids);
    }


    for word in words.iter_mut() {
        let mut changed = true;
        
        while changed {
            changed = false;
            
           
            let lowest_rank_pair = word.windows(2)
                .filter_map(|pair| {
                    let key = (pair[0], pair[1]);
                    self.ranks.get(&key).map(|&rank| (key, rank))
                })
                .min_by_key(|&(_, rank)| rank);

            if let Some((pair, _)) = lowest_rank_pair {
                if let Some(&merged_id) = self.pairs_to_merge.get(&pair) {
                    
                    let mut i = 0;
                    while i < word.len().saturating_sub(1) {
                        if word[i] == pair.0 && word[i + 1] == pair.1 {
                            word.remove(i + 1);
                            word[i] = merged_id;
                            changed = true;  
                           
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }

        for id in word.iter() {
            encoder_object.ids.push(*id);
            if let Some(token) = self.token_table.get_token(*id) {
                encoder_object.tokens.push(token.to_string());
            }
        }
    }

    encoder_object
}

pub fn display(& self){
        
        println!("{:?}",self.token_table.tokens());
        

        println!("{}",self.token_table.get_len());
    }

pub fn get_id_by_token(&self,token:String){
    let id=self.token_table.get_id(&token);
    println!("{}",id);
    
}
pub fn get_token_by_id(&self,id:u32){
    if let Some(token)=self.token_table.get_token(id){
        println!("{}",token.to_string());
    }
    else{
        print!("No token found for this id");
    }
    
    
}
pub fn get_merge_pair_from_pairs_to_merge_and_ranks(&self,tuple:(u32,u32)){
    if let Some(rank)=self.ranks.get(&tuple){
        println!("{}",rank);
    }
    else{
        println!("no rank found for this");
    }

    if let Some(pair)=self.pairs_to_merge.get_key_value(&tuple){
        let key=pair.0;
        let id=pair.1;
        println!("{:?}",key);
        println!("{}",id);
    }
    else{
        println!("no pair and id found for this tuple in pairs to merge");
    }

    if self.merge_rules.contains(&tuple){
        println!("pair found in merge rules")
    }
    else{
        println!("pair not found in merge rules")
    }
}

    
}




pub fn build_ranks_and_pairs_to_merge(merge_rules: &Vec<(u32,u32)>,table:&TokenTable)-> (HashMap<(u32, u32), usize>, HashMap<(u32, u32), u32>){
    let mut ranks: HashMap<(u32,u32),usize>=HashMap::new();
    let mut pairs_to_merge: HashMap<(u32,u32),u32>=HashMap::new();
    for (index ,pair) in merge_rules.iter().enumerate(){
        ranks.insert(*pair, index);
        let token1=table.get_token(pair.0).expect("token not found");
        let token2=table.get_token(pair.1).expect("token not found");
        let merged_pair=format!("{}{}", token1,token2);
        let merged_pair_id=table.get_id(&merged_pair);
        pairs_to_merge.insert(*pair, merged_pair_id);
    }
    (ranks,pairs_to_merge)
}

pub fn vocab_builder(pretokenized_text:&[String],vocab:&mut Vec<Word>, table: &mut TokenTable,pair_occurrences: &mut HashMap<(u32, u32), HashSet<(usize, usize)>> )   {
    
    let mut word_index_map:HashMap<Vec<u32>,usize>=HashMap::new();

    for (indx,word )in pretokenized_text.iter().enumerate(){
        let mut char_seq:Vec<u32> = word.chars().map(|c| {let ch=c.to_string(); table.get_or_insert_id(&ch)}).collect();
        let end_of_word_id=table.get_or_insert_id("</w>");
        char_seq.push(end_of_word_id);

        if let Some(&existing_indx)=word_index_map.get(&char_seq){
            vocab[existing_indx].count +=1;
        }
        else {
            let vocab_index=vocab.len();
            vocab.push(Word{tokens:char_seq.clone(),count:1});
            word_index_map.insert(char_seq.clone(), vocab_index);

            for i in 0..char_seq.len().saturating_sub(1){
                let pair=(char_seq[i],char_seq[i+1]);
                pair_occurrences.entry(pair).or_insert_with(HashSet::new).insert((vocab_index,i));
            }
        }
       

    }


}


pub fn vocab_builder2(
    pretokenized_text: &[String],
    vocab: &mut Vec<Word>,
    table: &mut TokenTable,
    pair_occurrences: &mut HashMap<(u32, u32), HashSet<(usize, usize)>>
) {
   
    let mut char_to_id: HashMap<char, u32> = HashMap::new();
    for word in pretokenized_text.iter() {
        for c in word.chars() {
            let s = c.to_string();
            char_to_id.entry(c).or_insert_with(|| table.get_or_insert_id(&s));
        }
    }

    let end_of_word_id = table.get_or_insert_id("</w>");

    let raw_words: Vec<Vec<u32>> = pretokenized_text
        .par_iter()
        .map(|word| {
            let mut char_seq: Vec<u32> = word
                .chars()
                .map(|c| char_to_id[&c])
                .collect();
            char_seq.push(end_of_word_id);
            char_seq
        })
        .collect();

    
    let mut word_index_map: HashMap<Vec<u32>, usize> = HashMap::new();

    for char_seq in raw_words {
        if let Some(&existing_indx) = word_index_map.get(&char_seq) {
            vocab[existing_indx].count += 1;
        } else {
            let vocab_index = vocab.len();
            vocab.push(Word {
                tokens: char_seq.clone(),
                count: 1,
            });
            word_index_map.insert(char_seq.clone(), vocab_index);

            for i in 0..char_seq.len().saturating_sub(1) {
                let pair = (char_seq[i], char_seq[i + 1]);
                pair_occurrences
                    .entry(pair)
                    .or_insert_with(HashSet::new)
                    .insert((vocab_index, i));
            }
        }
    }
}




pub fn build_initial_pair_stats(vocab:&Vec<Word>) -> (HashMap<(u32,u32),usize> , BinaryHeap<(usize,(u32,u32))>){
    let mut pair_freq: HashMap<(u32,u32),usize>=HashMap::new();
    let mut heap: BinaryHeap<(usize,(u32,u32))>=BinaryHeap::new();
    
    for (word) in vocab{
         if word.tokens.len() <2{
            continue;
        }
        for pair in word.tokens.windows(2){
            let key=(pair[0],pair[1]);
            *pair_freq.entry(key).or_insert(0) +=word.count;
  
        }

    }

    for (&pair,&freq) in &pair_freq{
        heap.push((freq,pair));

    }

    (pair_freq,heap)
    
}

#[derive(Debug)]
enum TokenizerError{
    EmptyVocab,
}



fn safe_subtraction(pair_freq:&mut HashMap<(u32,u32),usize>,key:(u32,u32),count:usize){
    if let Some(val)=pair_freq.get_mut(&key){
        *val=val.saturating_sub(count);
    }
}


pub fn update_pair_stats_after_merge(vocab:& Vec<Word>,pair_freq:&mut HashMap<(u32,u32),usize>,heap:&mut BinaryHeap<(usize,(u32,u32))> ,max_pair:& (u32,u32),max_pair_id:u32, pair_occurrences: & HashMap<(u32, u32), HashSet<(usize, usize)>> ){
    if let Some(positions)=pair_occurrences.get(max_pair){
        
        for &(vocab_indx,pos) in positions{
            let word=& vocab[vocab_indx];
            
            if pos >= word.tokens.len() - 1 {
                continue;
            }

            if word.tokens[pos] != max_pair.0 || word.tokens[pos + 1] != max_pair.1 {
                continue;
            }


            if pos>0{
                    let left= (word.tokens[pos-1],max_pair.0);
                    safe_subtraction(pair_freq, left, word.count);
                    heap.push((*pair_freq.get(&left).unwrap_or(&0),left));

                    let new_left=(word.tokens[pos-1],max_pair_id);
                    *pair_freq.entry(new_left).or_insert(0) +=word.count;
                    heap.push((*pair_freq.get(&new_left).unwrap(),new_left));

                }
                if pos+2< word.tokens.len(){
                    let right =(max_pair.1,word.tokens[pos+2]);
                    safe_subtraction(pair_freq, right, word.count);
                    heap.push((*pair_freq.get(&right).unwrap_or(&0),right));

                    let new_right =(max_pair_id,word.tokens[pos+2]);
                    *pair_freq.entry(new_right).or_insert(0) +=word.count;
                    heap.push((*pair_freq.get(&new_right).unwrap(),new_right));

                    
                }

        }
        pair_freq.remove(max_pair);
    } 
}


pub fn replace_occurance_with_merged_pair(vocab:&mut Vec<Word>,max_pair:& (u32,u32),max_pair_id:u32, pair_occurrences: &mut HashMap<(u32, u32), HashSet<(usize, usize)>>){
    if let Some(positions)=pair_occurrences.get(max_pair){
        let mut Positions_vec:Vec<(usize,usize)>=positions.iter().cloned().collect();
        Positions_vec.sort_by_key(|&(_, pos)| pos);
        for (vocab_indx,pos) in Positions_vec{
            let word=&mut vocab[vocab_indx];

            // if pos>word.tokens.len()-1 {
            //     continue;
            // }
            if pos + 1 >= word.tokens.len() {
                continue;
            }

            if pos>0{
                let left=(word.tokens[pos-1],word.tokens[pos]);
                if let Some(set)=pair_occurrences.get_mut(&left){
                    set.remove(&(vocab_indx,pos-1));
                }
            }
            
            if pos+2<word.tokens.len(){
                let right=(word.tokens[pos+1],word.tokens[pos+2]);
                if let Some(set)=pair_occurrences.get_mut(&right){
                    set.remove(&(vocab_indx,pos+1));
                }
            }

            if let Some(pr)=pair_occurrences.get_mut(max_pair){
                pr.remove(&(vocab_indx,pos));
            }

           

            if word.tokens[pos] != max_pair.0 || word.tokens[pos + 1] != max_pair.1 {
                continue;
            }

            word.tokens.splice(pos..=pos+1, [max_pair_id]);
            

            if pos > 0 && pos < word.tokens.len()  {
                let new_left = (word.tokens[pos - 1], max_pair_id);
                pair_occurrences.entry(new_left).or_default().insert((vocab_indx, pos - 1));
            }

            if pos +1 < word.tokens.len()  {
                let new_right = (max_pair_id, word.tokens[pos + 1]);
                pair_occurrences.entry(new_right).or_default().insert((vocab_indx, pos));
            }

           
        }

    }
    pair_occurrences.remove(max_pair);
}