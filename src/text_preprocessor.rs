use regex::Regex;
use log::debug;

/// Text preprocessing component for semantic-aware chunking
/// 
/// Handles cleaning and normalization before semantic segmentation
pub struct TextPreprocessor {
    excessive_newlines_regex: Regex,
    whitespace_cleanup_regex: Regex,
    control_chars_regex: Regex,
}

impl TextPreprocessor {
    pub fn new() -> Self {
        Self {
            // Your Python regex: r'\n\s*\n\s*\n+' -> '\n\n'
            excessive_newlines_regex: Regex::new(r"\n\s*\n\s*\n+").expect("Invalid newlines regex"),
            // General whitespace cleanup
            whitespace_cleanup_regex: Regex::new(r"[ \t]+").expect("Invalid whitespace regex"),
            // Remove control characters but preserve newlines/tabs
            control_chars_regex: Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").expect("Invalid control chars regex"),
        }
    }
    
    /// Clean and normalize text before semantic chunking
    /// 
    /// Steps:
    /// 1. Remove control characters
    /// 2. Normalize excessive newlines (your Python pattern)
    /// 3. Clean excessive spaces/tabs
    /// 4. Trim edges
    pub fn preprocess(&self, text: &str) -> String {
        debug!("Preprocessing text: {} characters", text.len());
        
        // Step 1: Remove control characters (preserve \n, \t)
        let no_control = self.control_chars_regex.replace_all(text, "");
        
        // Step 2: Apply your Python newline cleaning: \n\s*\n\s*\n+ -> \n\n
        let clean_newlines = self.excessive_newlines_regex.replace_all(&no_control, "\n\n");
        
        // Step 3: Clean excessive spaces/tabs (preserve single spaces)
        let clean_spaces = self.whitespace_cleanup_regex.replace_all(&clean_newlines, " ");
        
        // Step 4: Trim and normalize
        let result = clean_spaces.trim().to_string();
        
        debug!("Preprocessed text: {} -> {} characters", text.len(), result.len());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excessive_newlines() {
        let preprocessor = TextPreprocessor::new();
        let input = "Line 1\n\n\n\nLine 2\n \n \n\nLine 3";
        let result = preprocessor.preprocess(input);
        assert_eq!(result, "Line 1\n\nLine 2\n\nLine 3");
    }

    #[test]
    fn test_whitespace_cleanup() {
        let preprocessor = TextPreprocessor::new();
        let input = "Word1    Word2\t\t\tWord3   Word4";
        let result = preprocessor.preprocess(input);
        assert_eq!(result, "Word1 Word2 Word3 Word4");
    }
}
