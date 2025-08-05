use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;


pub struct TokenTable{
    id_to_token:Vec<Arc<str>>,
    token_to_id: HashMap<Arc<str>,u32>
}

impl TokenTable{
    pub fn new() -> Self{
        TokenTable{
            id_to_token:Vec::new(),
            token_to_id:HashMap::new()
        }
    }

    pub fn get_or_insert_id(&mut self, token:&str)->u32{
        if let Some(&id)= self.token_to_id.get(token){
            return id
        }
        let rc_token: Arc<str> = Arc::from(token);
        let id=self.id_to_token.len() as u32;
        self.id_to_token.push(Arc::clone(&rc_token));
        self.token_to_id.insert(rc_token,id);
        id
    }

    pub fn get_id(&self, token:&str)->Option<u32>{
        return  self.token_to_id.get(token).copied();

    }

    pub fn get_token(&self, id:u32) -> Option<&Arc<str>>{
        return self.id_to_token.get(id as usize);
    }

    pub fn get_len(&self) -> usize{
        self.id_to_token.len()
    }

    pub fn tokens(&self) -> &[Arc<str>]{
        &self.id_to_token
    }
}

