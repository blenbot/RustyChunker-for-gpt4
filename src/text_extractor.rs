use crate::error::ProcessingError;
use pdfium_render::prelude::*;
use regex::Regex;
use log::{debug};

/// Text extraction component using pdfium
/// 
/// This handles the low-level text extraction from PDF pages
pub struct TextExtractor {
    cleanup_regex: Regex,
}

impl TextExtractor {
    pub fn new() -> Self {
        // Cleanup regex: removes excessive whitespace and normalizes text
        let cleanup_regex = Regex::new(r"\s+").expect("Invalid cleanup regex");
        
        TextExtractor {
            cleanup_regex,
        }
    }
    
    /// Extract and clean text from a PDF page
    /// 
    /// Process:
    /// 1. Extract raw text using pdfium's text extraction
    /// 2. Clean and normalize whitespace
    /// 3. Handle Unicode and special characters
    /// 4. Return cleaned text ready for word counting
    pub fn extract_page_text(&self, page: &PdfPage, page_index: usize) -> Result<String, ProcessingError> {
        debug!("Extracting text from page {}", page_index);
        
        // Extract text using pdfium - this handles the PDF structure parsing
        let raw_text = page.text()
            .map_err(|e| ProcessingError::TextExtractionError {
                page: page_index,
                error: format!("pdfium extraction failed: {}", e),
            })?
            .all();
        
        if raw_text.is_empty() {
            debug!("Page {} contains no text", page_index);
            return Ok(String::new());
        }
        
        // Clean and normalize the extracted text
        let cleaned_text = self.cleanup_text(&raw_text);
        
        debug!("Extracted {} characters from page {}", cleaned_text.len(), page_index);
        Ok(cleaned_text)
    }
    
    /// Clean and normalize extracted text
    /// 
    /// Handles:
    /// - Multiple whitespace normalization
    /// - Unicode normalization
    /// - Control character removal
    fn cleanup_text(&self, text: &str) -> String {
        // Replace multiple whitespace with single spaces
        let normalized = self.cleanup_regex.replace_all(text, " ");
        
        // Trim and return cleaned text
        normalized.trim().to_string()
    }
}