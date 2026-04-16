import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from pdfminer.high_level import extract_text as pdfminer_extract_text

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index.bin"
FAISS_STORE_PATH = "faiss_store.pkl"
TFIDF_PATH = "tfidf.pkl"


DATA_DIR = "data"

def load_pdfs():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                path = os.path.join(root, fname)
                print(f"Loading PDF: {path}")
                text = hybrid_extract_text(path)
                print(f"  -> Final extracted length for {fname}: {len(text)} chars")

                docs.append({
                    "source": os.path.relpath(path, DATA_DIR),
                    "text": text
                })

    print(f"Total PDFs found: {len(docs)}")
    return docs



# def load_pdfs():
#     docs = []
#     for root, _, files in os.walk(DATA_DIR):
#         for fname in files:
#             if fname.lower().endswith(".pdf"):
#                 path = os.path.join(root, fname)
#                 print("Loading:", path)
#                 reader = PdfReader(path)
#                 text = ""
#                 for page in reader.pages:
#                     try:
#                         text += page.extract_text() or ""
#                     except Exception:
#                         continue
#                 docs.append({"source": os.path.relpath(path, DATA_DIR), "text": text})
#     print("Total PDFs loaded:", len(docs))
#     return docs


def extract_with_pypdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            except Exception as e:
                print(f"PyPDF error on page {i} in {path}: {e}")
        return text
    except Exception as e:
        print(f"PyPDF failed for {path}: {e}")
        return ""
    

def extract_with_pdfminer(path: str) -> str:
    try:
        text = pdfminer_extract_text(path)
        return text or ""
    except Exception as e:
        print(f"pdfminer failed for {path}: {e}")
        return ""

def extract_with_ocr(path: str) -> str:
    if not OCR_AVAILABLE:
        print(f"OCR not available, skipping for {path}")
        return ""

    try:
        reader = PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                # Render page to image via PyPDF (simple approach: use page.images if present)
                # NOTE: PyPDF doesn't render; for real OCR you'd use pdf2image.
                # Here we just show the structure; adapt if you add pdf2image.
                pass
            except Exception as e:
                print(f"OCR error on page {i} in {path}: {e}")
        return text
    except Exception as e:
        print(f"OCR failed for {path}: {e}")
        return ""


def hybrid_extract_text(path: str) -> str:
    print(f"  -> Trying PyPDF for {path}")
    text = extract_with_pypdf(path)
    if len(text.strip()) > 500:  # heuristic: enough text
        print(f"  ✅ PyPDF succeeded for {path} with {len(text)} chars")
        return text

    print(f"  ⚠️ PyPDF weak/empty for {path}, trying pdfminer...")
    text = extract_with_pdfminer(path)
    if len(text.strip()) > 500:
        print(f"  ✅ pdfminer succeeded for {path} with {len(text)} chars")
        return text

    if OCR_AVAILABLE:
        print(f"  ⚠️ pdfminer weak/empty for {path}, trying OCR...")
        text = extract_with_ocr(path)
        print(f"  OCR result for {path}: {len(text)} chars")
        return text

    print(f"  ❗ WARNING: All extractors weak/empty for {path}, got {len(text)} chars")
    return text


def chunk_text(text, chunk_size=600, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_corpus(docs):
    corpus = []
    meta = []
    for d in docs:
        chunks = chunk_text(d["text"])
        for c in chunks:
            corpus.append(c)
            meta.append({"source": d["source"], "text": c})
    return corpus, meta


def main():
    print("Loading PDFs...")
    docs = load_pdfs()

    print("Building corpus...")
    corpus, meta = build_corpus(docs)

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=20000)
    embeddings = vectorizer.fit_transform(corpus).toarray().astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving FAISS index and metadata...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_STORE_PATH, "wb") as f:
        pickle.dump(meta, f)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Done. Chunks stored:", len(meta))


if __name__ == "__main__":
    main()
