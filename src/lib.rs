use pyo3::prelude::*;
use pyo3::types::PyDict;

mod pdf_processor;
mod chunking;
mod parallel_processor;
mod text_extractor;
mod error;
mod tiktoken_core;
mod o200k_vocab;

use pdf_processor::PdfProcessor;

/// Python module initialization
/// This is the entry point that Maturin uses to create the Python extension
#[pymodule]
fn myrustchunker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging for debugging (optional)
    env_logger::init();
    
    // Register the main processing function
    m.add_function(wrap_pyfunction!(process_pdf, m)?)?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

/// Main Python-exposed function for processing PDFs
/// 
/// This function takes a PDF file path and returns chunk metadata as a list of dictionaries
/// Each dictionary contains: page, chunk_id, text, and source
/// 
/// Architecture:
/// 1. Load PDF using pdfium-render
/// 2. Extract text from all pages in parallel batches
/// 3. Apply chunking logic per page (300 words with 60 word overlap)
/// 4. Return structured metadata for Python consumption
#[pyfunction]
fn process_pdf(py: Python, pdf_path: String) -> PyResult<Vec<PyObject>> {
    // Create tokio runtime with correct API
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create runtime: {}", e)))?;
    
    rt.block_on(async {
        // Initialize the PDF processor with dynamic core detection
        let processor = PdfProcessor::new().await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Processor initialization failed: {}", e)))?;
        
        // Process the PDF and get chunk metadata
        let chunks = processor.process_pdf(&pdf_path).await
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("PDF processing failed: {}", e)))?;
        
        // Convert Rust structs to Python dictionaries
        let mut result = Vec::new();
        for chunk in chunks {
            let dict = PyDict::new(py);  // Changed from PyDict::new_bound to PyDict::new
            dict.set_item("page", chunk.page)?;
            dict.set_item("chunk_id", chunk.chunk_id)?;
            dict.set_item("text", chunk.text)?;
            dict.set_item("source", chunk.source)?;
            dict.set_item("token_count", chunk.token_count)?;  // Add token count to output
            result.push(dict.into());
        }
        
        Ok(result)
    })
}