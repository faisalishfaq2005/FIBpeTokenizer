#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "python")]
use crate::{bpe_tokenizer::{BpeTokenizer, TokenizerError, Encoder, TokenType}, PreTokenization, SpecialTokenRemovalMethod};

#[cfg(feature = "python")]
#[pyclass(name = "BpeTokenizer")]
pub struct PyBpeTokenizer {
    inner: BpeTokenizer
}

#[cfg(feature = "python")]
#[pyclass(name = "PreTokenization")]
#[derive(Clone)]
pub struct PyPreTokenization {
    inner: PreTokenization
}

#[cfg(feature = "python")]
#[pyclass(name = "SpecialTokenRemovalMethod")]
#[derive(Clone)]
pub struct PySpecialTokenRemovalMethod {
    inner: SpecialTokenRemovalMethod
}

#[cfg(feature = "python")]
#[pyclass(name = "Encoder")]
pub struct PyEncoder {
    inner: Encoder
}

#[cfg(feature = "python")]
#[pyclass(name = "TokenType")]
#[derive(Clone)]
pub enum PyTokenType {
    WORD,
    SUBWORD,
    SPECIALTOKEN,
}

#[cfg(feature = "python")]
impl From<TokenType> for PyTokenType {
    fn from(token_type: TokenType) -> Self {
        match token_type {
            TokenType::WORD => PyTokenType::WORD,
            TokenType::SUBWORD => PyTokenType::SUBWORD,
            TokenType::SPECIALTOKEN => PyTokenType::SPECIALTOKEN,
        }
    }
}

#[cfg(feature = "python")]
impl From<TokenizerError> for PyErr {
    fn from(err: TokenizerError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyPreTokenization {
    #[staticmethod]
    fn whitespace() -> Self {
        PyPreTokenization {
            inner: PreTokenization::Whitespace
        }
    }

    #[staticmethod]
    fn punctuation() -> Self {
        PyPreTokenization {
            inner: PreTokenization::Punctuation
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PySpecialTokenRemovalMethod {
    #[staticmethod]
    fn simple() -> Self {
        PySpecialTokenRemovalMethod {
            inner: SpecialTokenRemovalMethod::Simple
        }
    }

    #[staticmethod]
    fn aho_corasick() -> Self {
        PySpecialTokenRemovalMethod {
            inner: SpecialTokenRemovalMethod::AhoCorasick
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyEncoder {
    #[getter]
    fn original_text(&self) -> String {
        self.inner.original_text.clone()
    }

    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.inner.ids.clone()
    }

    #[getter]
    fn token_types(&self) -> Vec<PyTokenType> {
        self.inner.token_types.iter().map(|t| t.clone().into()).collect()
    }

    fn get_token_type(&self, token: &str) -> PyResult<PyTokenType> {
        match self.inner.get_token_type(token) {
            Ok(token_type) => Ok(token_type.into()),
            Err(e) => Err(e.into())
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBpeTokenizer {
    #[new]
    #[pyo3(signature = (input_path, target_vocab_size, pretokenization_type, special_tokens=Vec::new(), special_token_removal_method=None, save_model=false, output_dir=None))]
    fn new(
        input_path: &str,
        target_vocab_size: usize,
        pretokenization_type: &PyPreTokenization,
        special_tokens: Vec<String>,
        special_token_removal_method: Option<&PySpecialTokenRemovalMethod>,
        save_model: bool,
        output_dir: Option<&str>
    ) -> Self {
        let removal_method = special_token_removal_method
            .map(|m| m.inner.clone())
            .unwrap_or(SpecialTokenRemovalMethod::Simple);

        PyBpeTokenizer {
            inner: BpeTokenizer::new(
                input_path,
                target_vocab_size,
                pretokenization_type.inner.clone(),
                special_tokens,
                removal_method,
                save_model,
                output_dir
            )
        }
    }

    #[staticmethod]
    fn from_pretrained(files_path: &str) -> Self {
        PyBpeTokenizer {
            inner: BpeTokenizer::new_from_pretrained(files_path) 
        }
    }

    fn train(&mut self) -> PyResult<()> {
        self.inner.train().map_err(|e| e.into())
    }

    fn encode(&self, text: &str) -> PyResult<PyEncoder> {
        match self.inner.encode(text) {
            Ok(encoder) => Ok(PyEncoder { inner: encoder }),
            Err(e) => Err(e.into())
        }
    }

    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(|e| e.into())
    }

    fn get_id_by_token(&self, token: String) -> PyResult<u32> {
        self.inner.get_id_by_token(token).map_err(|e| e.into())
    }

    fn get_token_by_id(&self, id: u32) -> PyResult<String> {
        self.inner.get_token_by_id(id).map_err(|e| e.into())
    }

    fn __repr__(&self) -> String {
        format!("BpeTokenizer(vocab_size={})", self.inner.token_table.get_len())
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn fibpetokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBpeTokenizer>()?;
    m.add_class::<PyPreTokenization>()?;
    m.add_class::<PySpecialTokenRemovalMethod>()?;
    m.add_class::<PyEncoder>()?;
    m.add_class::<PyTokenType>()?;
    Ok(())
}

