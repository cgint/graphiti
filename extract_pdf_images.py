#!/usr/bin/env python3
"""Extract PDF pages as images and text - one image and text file per page."""

import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print('Error: PyMuPDF is required. Install it with: uv pip install pymupdf')
    sys.exit(1)


def extract_pdf_to_images_and_text(pdf_path: str, output_dir: str = None):
    """Extract each page of a PDF as a separate image file and text file."""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        print(f'Error: PDF file not found: {pdf_path}')
        sys.exit(1)
    
    # Set output directory to same location as PDF if not specified
    if output_dir is None:
        output_dir = pdf_path.parent / f'{pdf_path.stem}_images'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Open PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    
    print(f'Extracting {total_pages} pages from {pdf_path.name}...')
    
    # Extract text from all pages into one file
    all_text = []
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        text = page.get_text()
        all_text.append(f'--- Page {page_num + 1} ---\n\n{text}\n\n')
    
    # Save all text to a single file
    text_path = output_dir / f'{pdf_path.stem}.txt'
    text_path.write_text(''.join(all_text), encoding='utf-8')
    print(f'  Saved text: {text_path.name}')
    
    # Extract each page as an image
    for page_num in range(total_pages):
        page = pdf_document[page_num]
        
        # Render page to image (pixmap) at 2x resolution for better quality
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for higher resolution
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = output_dir / f'page_{page_num + 1:03d}.png'
        pix.save(image_path)
        print(f'  Saved image: {image_path.name}')
    
    pdf_document.close()
    print(f'\nDone! Extracted {total_pages} images and 1 text file to: {output_dir}')


if __name__ == '__main__':
    pdf_file = 'A TEMPORAL KNOWLEDGE GRAPH FOR AGENT MEMORY.pdf'
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    
    extract_pdf_to_images_and_text(pdf_file)

