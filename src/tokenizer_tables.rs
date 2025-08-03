use std::collections::HashMap;
use std::rc::Rc;


pub struct TokenTable{
    id_to_token:Vec<Rc<str>>,
    token_to_id: HashMap<Rc<str>,u32>
}

impl TokenTable{
    pub fn new() -> Self{
        TokenTable{
            id_to_token:Vec::new(),
            token_to_id:HashMap::new()
        }
    }

    pub fn get_or_insert_id(&mut self, token:&str)->u32{
        if let some(&id)= self.token_to_id.get(token){
            return id
        }
        let rc_token: Rc<str> = Rc::from(token);
        let id=self.id_to_token.len() as u32;
        self.id_to_token.push(Rc::clone(&rc_token));
        self.token_to_id.insert(rc_token,id);
        id
    }

    pub fn get_id(&self, token:&str)->Option<u32>{
        self.token_to_id.get(token).copied();

    }

    pub fn get_token(&self, id:u32) -> Option<&Rc<str>>{
        self.id_to_token.get(id as usize);
    }

    pub fn get_len(&self) -> usize{
        self.id_to_token.len()
    }

    pub fn tokens(&self) -> &[Rc<str>]{
        &self.id_to_token
    }
}

