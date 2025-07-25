use std::{collections::{HashMap, HashSet}, fmt::format, usize};
fn main() {
    let raw_text:&str="i am going there and low lowest low there going what doing in there";
    let tokens:Vec<String>= raw_text.split_whitespace().map(|s| s.to_string()).collect();
    let target_vocab_size=50;

    let mut final_vocabulary:HashSet<String>=HashSet::new();
    let mut vocab:HashMap<Vec<String>,usize>=HashMap::new();
    let mut merge_rules:Vec<(String,String)>=Vec::new();

    vocab_builder(&tokens, &mut vocab);
    add_initial_unique_ch_in_final_vocab(&mut final_vocabulary, &vocab);
    let mut vocab_len=final_vocabulary.len();
    println!("{}" ,vocab_len);
    for ch in & final_vocabulary{
        println!("{}",ch);
    }

    for (token, count) in &vocab {
        println!("{:?}: {}", token, count);
    }
    while  final_vocabulary.len()<target_vocab_size {
    
        
        let max_pair: Option<(String, String)>= record_most_frequent_adjacent_pair(&vocab);
        if max_pair.is_none(){
            print!("no more pairs to merge");
            break;
        }

        let max_pair_tuple:(String,String)=max_pair.unwrap();
        let merged_pair=format!("{}{}", max_pair_tuple.0,max_pair_tuple.1);
        if final_vocabulary.insert(merged_pair.clone()){
            merge_rules.push(max_pair_tuple.clone());
        }
        println!("{:?}",max_pair_tuple);
        //let example_pair:(String,String)=("g".to_string(),"</w>".to_string());
        vocab=replace_occurance_with_merged_pair(& vocab, &max_pair_tuple);

    }

    print!("final_vocabulary");
    for ch in & final_vocabulary{
        println!("{}",ch);
    }
    
    print!("now printing merged vocab");
    print!("");
    for (token, count) in &vocab {
        println!("{:?}: {}", token, count);
    }

    print!("merge rules");
    for rule in merge_rules{
        println!("{:?}",rule);
    }


}

fn add_initial_unique_ch_in_final_vocab(final_vocabulary: &mut HashSet<String>,vocab:& HashMap<Vec<String>,usize>){
    for word in vocab.keys(){
        for token in word{
            final_vocabulary.insert(token.clone());
        }
    }
}



fn vocab_builder(pretokenized_text:&Vec<String>,vocab:&mut HashMap<Vec<String>,usize>)   {
    
    for word in pretokenized_text{
        let mut char_seq:Vec<String> = word.chars().map(|c| c.to_string()).collect();
        char_seq.push("</w>".to_string());
        *vocab.entry(char_seq).or_insert(0) +=1;

    }


}

#[derive(Debug)]
enum TokenizerError{
    EmptyVocab,
}

fn record_most_frequent_adjacent_pair(vocab:&HashMap<Vec<String>,usize>) -> Option<(String,String)>{
    let mut adjacent_pairs_frequency_hashmap: HashMap<(String,String),usize>=HashMap::new();
    
    for  (word,count) in vocab {
        if word.len() <2{
            continue;
        }
        for pair in word.windows(2)
        {
            let key = (pair[0].clone(),pair[1].clone());
            *adjacent_pairs_frequency_hashmap.entry(key).or_insert(0) +=count;

        }
    }
    for (tuple,count) in &adjacent_pairs_frequency_hashmap{
        println!("{:?}: {}", tuple, count);
    }
    let max_pair=adjacent_pairs_frequency_hashmap.into_iter().max_by_key(|entry| entry.1).map(|(pair,_count)| pair);
    max_pair
}

fn replace_occurance_with_merged_pair(vocab:& HashMap<Vec<String>,usize>,max_pair:& (String,String))-> HashMap<Vec<String>,usize> {
    let mut updated_vocab:HashMap<Vec<String>,usize>=HashMap::new();

    for (word,count) in vocab{
        if word.len()<2{
            continue;
        }
        let mut new_word:Vec<String>=Vec::new();
        let mut i=0;
        while i<word.len() {
            if i<word.len()-1 && word[i]==max_pair.0 && word[i+1]==max_pair.1{
                let merged=format!("{}{}",word[i],word[i+1]);
                new_word.push(merged);
                i+=2;
            }
            else{
                new_word.push(word[i].clone());
                i+=1;
            }
                
        }
        *updated_vocab.entry(new_word).or_insert(0) += count;

    }
    updated_vocab
}

