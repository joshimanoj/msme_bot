import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import logging
import gdown
import tempfile
import hashlib
from utils import get_embeddings  # Ensure this is available

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_faiss_index(google_drive_file_id="1MQFFB-TEmKD8ToAyiQk49lQPQDTfedEp", output_path="faiss_index", version_file="faiss_version.txt"):
    """
    Precompute FAISS index for scheme_db.xlsx and save it to disk.

    Args:
        google_drive_file_id (str): Google Drive file ID for scheme_db.xlsx.
        output_path (str): Directory to save the FAISS index.
        version_file (str): File to save the Excel file hash.
    """
    # Download the Excel file
    download_url = f"https://docs.google.com/spreadsheets/d/{google_drive_file_id}/export?format=xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
        temp_file_path = temp_file.name
        logger.info(f"Downloading Google Sheet to {temp_file_path}")
        try:
            gdown.download(download_url, temp_file_path, quiet=False)
            logger.info("Download completed")
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise

        # Compute file hash
        with open(temp_file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        logger.info(f"Computed file hash: {file_hash}")

    # Read Excel file
    try:
        df = pd.read_excel(temp_file_path)
        logger.info(f"Excel file loaded. Rows: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read Excel file: {str(e)}")
        raise
    finally:
        os.unlink(temp_file_path)
        logger.info(f"Temporary file {temp_file_path} deleted")

    # Split data into two chunks
    total_rows = len(df)
    midpoint = total_rows // 2
    chunk1 = df.iloc[:midpoint]
    chunk2 = df.iloc[midpoint:]
    logger.info(f"Split data: Chunk 1 ({len(chunk1)} rows), Chunk 2 ({len(chunk2)} rows)")

    # Process chunks into documents
    def process_chunk(chunk):
        documents = []
        for _, row in chunk.iterrows():
            content_parts = []
            relevant_columns = [
                "name", "applicability", "type- SCH/DOC", "service type",
                "scheme type", "description", "objective(Eligibility)",
                "application method", "process", "benefit value description",
                "benefit amount (description)", "tags", "beneficiary type"
            ]
            for col in relevant_columns:
                if col in row and pd.notna(row[col]):
                    clean_col = col.replace('(', ' ').replace(')', '')
                    content_parts.append(f"{clean_col}: {row[col]}")
            content = "\n".join(content_parts)
            metadata = {
                "guid": row["guid"] if pd.notna(row["guid"]) else "",
                "name": row["name"] if pd.notna(row["name"]) else ""
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents

    # Create FAISS vector stores
    embeddings = get_embeddings()
    documents1 = process_chunk(chunk1)
    logger.info(f"Created {len(documents1)} documents from Chunk 1")
    vector_store1 = FAISS.from_documents(documents1, embeddings)
    logger.info(f"FAISS vector store for Chunk 1 created with {vector_store1.index.ntotal} documents")

    documents2 = process_chunk(chunk2)
    logger.info(f"Created {len(documents2)} documents from Chunk 2")
    vector_store2 = FAISS.from_documents(documents2, embeddings)
    logger.info(f"FAISS vector store for Chunk 2 created with {vector_store2.index.ntotal} documents")

    # Merge vector stores
    vector_store1.merge_from(vector_store2)
    logger.info(f"Combined FAISS vector store created with {vector_store1.index.ntotal} documents")

    # Save the vector store and version
    try:
        os.makedirs(output_path, exist_ok=True)
        vector_store1.save_local(output_path)
        logger.info(f"Saved FAISS vector store to {output_path}")
        with open(version_file, "w") as f:
            f.write(file_hash)
        logger.info(f"Saved file hash to {version_file}")
    except Exception as e:
        logger.error(f"Failed to save FAISS vector store or version: {str(e)}")
        raise

if __name__ == "__main__":
    precompute_faiss_index()