/// Real o200k_base vocabulary loader from tiktoken data file
/// 
/// This module loads the actual OpenAI o200k_base vocabulary from the tiktoken file
/// format, providing 100% compatibility with OpenAI's tokenization.

use rustc_hash::FxHashMap as HashMap;
use crate::tiktoken_core::Rank;
use crate::error::ProcessingError;
use std::str::FromStr;
use log::info;
use base64::{Engine as _, engine::general_purpose};

/// The actual o200k_base regex pattern used by OpenAI
pub const O200K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Load the real o200k_base encoder vocabulary from the tiktoken file
/// 
/// Format: Each line contains "base64_token rank"
/// Example: "IQ== 0" means token with bytes [33] has rank 0
pub fn load_o200k_base_encoder() -> Result<HashMap<Vec<u8>, Rank>, ProcessingError> {
    info!("Loading real o200k_base vocabulary from tiktoken file...");
    
    // Include the tiktoken file at compile time
    let tiktoken_data = include_str!("../o200k_base.tiktoken");
    
    let mut encoder = HashMap::default();
    let mut line_count = 0;
    let mut error_count = 0;
    
    for line in tiktoken_data.lines() {
        line_count += 1;
        
        if line.trim().is_empty() {
            continue;
        }
        
        // Parse line format: "base64_token rank"
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            error_count += 1;
            if error_count <= 10 {  // Only log first 10 errors to avoid spam
                eprintln!("Warning: Invalid line format at line {}: '{}'", line_count, line);
            }
            continue;
        }
        
        let base64_token = parts[0];
        let rank_str = parts[1];
        
        // Decode base64 token
        let token_bytes = match general_purpose::STANDARD.decode(base64_token) {
            Ok(bytes) => bytes,
            Err(e) => {
                error_count += 1;
                if error_count <= 10 {
                    eprintln!("Warning: Failed to decode base64 '{}' at line {}: {}", 
                             base64_token, line_count, e);
                }
                continue;
            }
        };
        
        // Parse rank
        let rank = match Rank::from_str(rank_str) {
            Ok(r) => r,
            Err(e) => {
                error_count += 1;
                if error_count <= 10 {
                    eprintln!("Warning: Failed to parse rank '{}' at line {}: {}", 
                             rank_str, line_count, e);
                }
                continue;
            }
        };
        
        // Insert into encoder map
        encoder.insert(token_bytes, rank);
    }
    
    if error_count > 10 {
        eprintln!("Warning: {} total parsing errors encountered (showing first 10)", error_count);
    }
    
    info!("Successfully loaded {} tokens from o200k_base.tiktoken", encoder.len());
    info!("Processed {} lines with {} errors", line_count, error_count);
    
    // Verify we have a reasonable number of tokens
    if encoder.len() < 100000 {
        return Err(ProcessingError::SystemError(
            format!("Loaded only {} tokens, expected ~200k. File may be corrupted.", encoder.len())
        ));
    }
    
    Ok(encoder)
}

/// Load special tokens for o200k_base
/// These are the standard OpenAI special tokens not included in the main vocabulary
pub fn load_o200k_base_special_tokens() -> HashMap<String, Rank> {
    let mut special_tokens = HashMap::default();
    
    // Standard OpenAI special tokens for o200k_base
    special_tokens.insert("<|endoftext|>".to_string(), 100257);
    special_tokens.insert("<|fim_prefix|>".to_string(), 100258);
    special_tokens.insert("<|fim_middle|>".to_string(), 100259);
    special_tokens.insert("<|fim_suffix|>".to_string(), 100260);
    special_tokens.insert("<|endofprompt|>".to_string(), 100276);
    
    // Additional special tokens that might be used
    special_tokens.insert("<|im_start|>".to_string(), 100264);
    special_tokens.insert("<|im_end|>".to_string(), 100265);
    special_tokens.insert("<|im_sep|>".to_string(), 100266);
    
    special_tokens
}

/// Utility function to decode a base64 token for debugging
#[allow(dead_code)]
pub fn decode_token_debug(base64_token: &str) -> Result<String, ProcessingError> {
    let bytes = general_purpose::STANDARD.decode(base64_token)
        .map_err(|e| ProcessingError::SystemError(format!("Base64 decode error: {}", e)))?;
    
    match String::from_utf8(bytes.clone()) {
        Ok(s) => Ok(format!("'{}'", s)),
        Err(_) => Ok(format!("bytes: {:?}", bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_o200k_base_encoder() {
        let encoder = load_o200k_base_encoder().unwrap();
        
        // Should have substantial vocabulary size
        assert!(encoder.len() > 100000, "Should load substantial vocabulary");
        
        // Should contain basic single-byte tokens
        assert!(encoder.contains_key(&vec![32])); // space
        assert!(encoder.contains_key(&vec![65])); // 'A'
        assert!(encoder.contains_key(&vec![97])); // 'a'
        
        println!("Loaded {} tokens", encoder.len());
    }
    
    #[test] 
    fn test_decode_token_display() {
        // Test with ASCII text
        let result = decode_token_debug("aGVsbG8=").unwrap(); // "hello" in base64
        assert!(result.contains("hello"));
        
        // Test with single byte
        let result = decode_token_debug("IQ==").unwrap(); // "!" in base64
        assert!(result.contains("!"));
    }
}
