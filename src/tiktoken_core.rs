use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use crate::error::ProcessingError;
use crate::o200k_vocab::{load_o200k_base_encoder, load_o200k_base_special_tokens, O200K_BASE_PATTERN};
use log::{error, info};

pub type Rank = u32;

/// Core BPE byte pair merge algorithm
fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

/// Convert text piece into tokens using BPE
pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

/// Thread-safe hash for thread-local regex storage
struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

const MAX_NUM_THREADS: usize = 128;

/// Core BPE tokenizer implementation for o200k_base
/// 
/// This is a fast, thread-safe implementation of OpenAI's tiktoken core
/// optimized specifically for chunking operations in PDF processing
#[derive(Clone)]
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    #[allow(dead_code)]
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    /// Get thread-local regex for performance
    fn _get_tl_regex(&self) -> &Regex {
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    /// Create new CoreBPE instance with real o200k_base configuration
    pub fn new_o200k_base() -> Result<Self, ProcessingError> {
        info!("Initializing tiktoken o200k_base tokenizer with real vocabulary...");
        
        // Load real o200k_base vocabulary and special tokens
        let encoder = load_o200k_base_encoder()?;
        let special_tokens_encoder = load_o200k_base_special_tokens();
        
        info!("Loaded {} regular tokens and {} special tokens", 
              encoder.len(), special_tokens_encoder.len());
        
        Self::new_internal(encoder, special_tokens_encoder, O200K_BASE_PATTERN)
    }

    /// Internal constructor
    fn new_internal(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, ProcessingError> {
        let regex = Regex::new(pattern)
            .map_err(|e| ProcessingError::SystemError(format!("Regex compilation failed: {}", e)))?;

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|"))
                .map_err(|e| ProcessingError::SystemError(format!("Special regex compilation failed: {}", e)))?
        };

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        if encoder.len() != decoder.len() {
            return Err(ProcessingError::SystemError(
                "Encoder and decoder must be of equal length".to_string()
            ));
        }

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex_tls: (0..MAX_NUM_THREADS).map(|_| regex.clone()).collect(),
            special_regex_tls: (0..MAX_NUM_THREADS)
                .map(|_| special_regex.clone())
                .collect(),
            sorted_token_bytes,
        })
    }

    /// Encode text to tokens (main entry point)
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        
        for mat in regex.find_iter(text) {
            let piece = match mat {
                Ok(m) => m.as_str().as_bytes(),
                Err(e) => {
                    error!("Regex match error: {}", e);
                    continue;
                }
            };
            
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    /// Encode with special tokens support
    #[allow(dead_code)]
    pub fn encode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> Result<Vec<Rank>, ProcessingError> {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = special_regex.find_from_pos(text, start_find)
                    .map_err(|e| ProcessingError::SystemError(format!("Special regex error: {}", e)))?;
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat_res in regex.find_iter(&text[start..end]) {
                let mat = mat_res
                    .map_err(|e| ProcessingError::SystemError(format!("Regex error: {}", e)))?;

                let piece = mat.as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                ret.extend(&byte_pair_encode(piece, &self.encoder));
            }

            match next_special {
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        Ok(ret)
    }

    /// Count tokens in text - optimized for chunking
    #[allow(dead_code)]
    pub fn count_tokens(&self, text: &str) -> Result<usize, ProcessingError> {
        let tokens = self.encode_ordinary(text);
        Ok(tokens.len())
    }

    /// Decode tokens back to text
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, ProcessingError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or_else(|| ProcessingError::SystemError(format!("Invalid token: {}", token)))?,
            };
            ret.extend(token_bytes);
        }
        
        String::from_utf8(ret)
            .map_err(|e| ProcessingError::SystemError(format!("UTF-8 decode error: {}", e)))
    }

    /// Get special tokens
    #[allow(dead_code)]
    pub fn special_tokens(&self) -> HashSet<&str> {
        self.special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encoding() {
        let bpe = CoreBPE::new_o200k_base().unwrap();
        let tokens = bpe.encode_ordinary("Hello world");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_token_counting() {
        let bpe = CoreBPE::new_o200k_base().unwrap();
        let count = bpe.count_tokens("This is a test").unwrap();
        assert!(count > 0);
    }
}
