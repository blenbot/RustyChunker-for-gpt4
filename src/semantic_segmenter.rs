use regex::Regex;
use log::debug;

/// Text segment with semantic boundaries
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub semantic_level: usize, // Which separator level created this segment
}

/// Semantic text segmenter using recursive separator strategy
/// 
/// Implements LangChain-style RecursiveTextSplitter approach:
/// 1. Try splitting by strongest semantic separators first
/// 2. Recursively split large segments with weaker separators
/// 3. Preserve semantic boundaries when possible
pub struct SemanticSegmenter {
    separators: Vec<SeparatorPattern>,
}

#[derive(Debug)]
struct SeparatorPattern {
    pattern: SeparatorType,
    level: usize,
    description: String,
}

#[derive(Debug)]
enum SeparatorType {
    Regex(Regex),
    Literal(String),
}

impl SemanticSegmenter {
    pub fn new() -> Self {
        let separators = vec![
            // Level 0: Paragraph breaks (strongest semantic boundary)
            SeparatorPattern {
                pattern: SeparatorType::Regex(Regex::new(r"\n\n+").expect("Invalid paragraph regex")),
                level: 0,
                description: "Paragraph breaks".to_string(),
            },
            
            // Level 1: Markdown headers
            SeparatorPattern {
                pattern: SeparatorType::Regex(Regex::new(r"\n### ").expect("Invalid h3 regex")),
                level: 1,
                description: "H3 headers".to_string(),
            },
            SeparatorPattern {
                pattern: SeparatorType::Regex(Regex::new(r"\n## ").expect("Invalid h2 regex")),
                level: 1,
                description: "H2 headers".to_string(),
            },
            SeparatorPattern {
                pattern: SeparatorType::Regex(Regex::new(r"\n# ").expect("Invalid h1 regex")),
                level: 1,
                description: "H1 headers".to_string(),
            },
            
            // Level 2: Single line breaks
            SeparatorPattern {
                pattern: SeparatorType::Literal("\n".to_string()),
                level: 2,
                description: "Line breaks".to_string(),
            },
            
            // Level 3: Sentence endings
            SeparatorPattern {
                pattern: SeparatorType::Literal(". ".to_string()),
                level: 3,
                description: "Period + space".to_string(),
            },
            SeparatorPattern {
                pattern: SeparatorType::Literal("? ".to_string()),
                level: 3,
                description: "Question + space".to_string(),
            },
            SeparatorPattern {
                pattern: SeparatorType::Literal("! ".to_string()),
                level: 3,
                description: "Exclamation + space".to_string(),
            },
            
            // Level 4: Punctuation
            SeparatorPattern {
                pattern: SeparatorType::Literal("; ".to_string()),
                level: 4,
                description: "Semicolon + space".to_string(),
            },
            SeparatorPattern {
                pattern: SeparatorType::Literal(", ".to_string()),
                level: 4,
                description: "Comma + space".to_string(),
            },
            
            // Level 5: Whitespace (weakest semantic boundary)
            SeparatorPattern {
                pattern: SeparatorType::Literal(" ".to_string()),
                level: 5,
                description: "Spaces".to_string(),
            },
        ];
        
        Self { separators }
    }
    
    /// Segment text using recursive separator strategy
    /// 
    /// Returns segments that respect semantic boundaries as much as possible
    pub fn segment(&self, text: &str, max_tokens: usize, tokenizer: &crate::tiktoken_core::CoreBPE) -> Vec<Segment> {
        debug!("Starting semantic segmentation: {} chars, max_tokens={}", text.len(), max_tokens);
        
        let initial_segment = Segment {
            text: text.to_string(),
            start_offset: 0,
            end_offset: text.len(),
            semantic_level: 0,
        };
        
        self.recursive_split(vec![initial_segment], max_tokens, tokenizer, 0)
    }
    
    /// Recursively split segments using separator hierarchy
    fn recursive_split(
        &self,
        segments: Vec<Segment>,
        max_tokens: usize,
        tokenizer: &crate::tiktoken_core::CoreBPE,
        separator_level: usize,
    ) -> Vec<Segment> {
        // Base case: no more separators to try
        if separator_level >= self.separators.len() {
            return segments;
        }
        
        let mut result = Vec::new();
        let mut needs_further_splitting = Vec::new();
        
        for segment in segments {
            let token_count = tokenizer.encode_ordinary(&segment.text).len();
            
            if token_count <= max_tokens {
                // Segment is small enough, keep it
                result.push(segment);
            } else {
                // Try to split this segment
                let split_segments = self.split_segment(&segment, separator_level);
                
                if split_segments.len() > 1 {
                    // Successfully split, add smaller segments for further processing
                    needs_further_splitting.extend(split_segments);
                } else {
                    // Couldn't split with this separator, try next level
                    needs_further_splitting.push(segment);
                }
            }
        }
        
        // If we have segments that need further splitting, recursively process them
        if !needs_further_splitting.is_empty() {
            let further_split = self.recursive_split(needs_further_splitting, max_tokens, tokenizer, separator_level + 1);
            result.extend(further_split);
        }
        
        result
    }
    
    /// Split a single segment using the specified separator
    fn split_segment(&self, segment: &Segment, separator_level: usize) -> Vec<Segment> {
        if separator_level >= self.separators.len() {
            return vec![segment.clone()];
        }
        
        let separator = &self.separators[separator_level];
        debug!("Trying to split segment with {}: {} chars", separator.description, segment.text.len());
        
        let splits = match &separator.pattern {
            SeparatorType::Regex(regex) => {
                self.split_by_regex(&segment.text, regex)
            }
            SeparatorType::Literal(literal) => {
                self.split_by_literal(&segment.text, literal)
            }
        };
        
        if splits.len() <= 1 {
            return vec![segment.clone()];
        }
        
        // Convert splits to segments with proper offsets
        let mut result = Vec::new();
        let mut current_offset = segment.start_offset;
        
        for split_text in splits {
            if !split_text.trim().is_empty() {
                let segment_end = current_offset + split_text.len();
                result.push(Segment {
                    text: split_text,
                    start_offset: current_offset,
                    end_offset: segment_end,
                    semantic_level: separator_level,
                });
                current_offset = segment_end;
            }
        }
        
        debug!("Split into {} segments using {}", result.len(), separator.description);
        result
    }
    
    /// Split text by regex pattern
    fn split_by_regex(&self, text: &str, regex: &Regex) -> Vec<String> {
        regex.split(text)
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty())
            .collect()
    }
    
    /// Split text by literal string
    fn split_by_literal(&self, text: &str, separator: &str) -> Vec<String> {
        text.split(separator)
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiktoken_core::CoreBPE;

    #[test]
    fn test_paragraph_splitting() {
        let segmenter = SemanticSegmenter::new();
        let tokenizer = CoreBPE::new_o200k_base().unwrap();
        
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let segments = segmenter.segment(text, 50, &tokenizer);
        
        assert!(segments.len() >= 3);
        assert!(segments[0].text.contains("First"));
        assert!(segments[1].text.contains("Second"));
    }
}
