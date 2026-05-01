"""
updateData ingests raw text files into the ChromaDB vector store.
Run this script whenever new source files are added to Test/RawTextFiles.
Already-indexed sources are skipped automatically (idempotent).

Pipeline:
  1. Load source files from disk
  2. Get or create the ChromaDB collection
  3. For each new source, split into overlapping chunks
  4. Compute sentence embeddings for each chunk
  5. Store each chunk in the vector store
"""

###########################
# Standard library imports
###########################
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


###########################
# Local application imports
###########################

from Controller.SearchData import cleanText, fetchDataFromFolder
from Controller.CategoryTagger import precomputeLabelEmbeddings, assignTags
from DatabaseRepository.TfidfRepository import TfidfRepository
from DatabaseRepository.VectorRepository import VectorRepository
from DatabaseService.RagService import RagService
from Dto.Source import Source
from Categories import CATEGORY_LABELS
from Config import Config


def main():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.Model.MaxTextChunkLength,
        chunk_overlap=Config.Model.OverLapChunkLength
    )

    sourceObjects: list[Source] = fetchDataFromFolder('./Test/RawTextFiles')

    client = chromadb.PersistentClient(path="./embeddings_db")
    try:
        # Create collection on first run; get existing collection on subsequent runs
        collection = client.create_collection(name="my_docs", embedding_function=Config.Model.EmbeddingFunction)
    except chromadb.errors.CollectionAlreadyExistsError:
        collection = client.get_collection(name="my_docs", embedding_function=Config.Model.EmbeddingFunction)

    vectorRepo = VectorRepository(collection)
    tfidfRepo = TfidfRepository(Config.Model.TfidfDbPath)
    ragService = RagService(vectorRepo, tfidfRepo)

    # Use the same model as the embedding function to keep vector space consistent
    model = SentenceTransformer('all-MiniLM-L6-v2')
    labelEmbeddings = precomputeLabelEmbeddings(model, CATEGORY_LABELS)

    for source in sourceObjects:
        # Skip sources already in the DB to keep the script idempotent
        if ragService.sourceExists(source.title):
            print(f"Skipping '{source.title}' (already indexed)")
            continue

        text = cleanText(source.content)
        chunkTextList = splitter.split_text(text)
        sentenceEmbeddings = model.encode(chunkTextList)

        # Average chunk embeddings to represent the whole source, then tag once
        source.tags = assignTags(sentenceEmbeddings.mean(axis=0), labelEmbeddings, CATEGORY_LABELS, Config.Model.TagTopN)

        for sentence, embedding in zip(chunkTextList, sentenceEmbeddings):
            source.content = sentence
            # ChromaDB expects plain lists, not numpy arrays
            ragService.addSources([source], embedding.tolist())


if __name__ == "__main__":
    main()
